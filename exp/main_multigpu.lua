--local data_path = "../../dataset/big-data/es/" --local vocab = 151989 --local base = 390
local data_path, vocab, base
local jinzhi = 1000 -- must jinzhi > base

local ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        print("warning: fbcunn not found. Falling back to cunn") 
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    LookupTable = nn.LookupTable
end
require('nngraph')
require('../base')
require('lfs')
local ptb = require('data')
local tds = require('tds')


local params = {batch_size=20,
seq_length=30, -- back to one element seq_length
layers=2,
decay=1.2,
rnn_size=1000,
dropout=0.5,
init_weight=0.04,
lr=1.0,
start_lr = 1.0,
vocab_size=vocab, -- need to fill later
vocab_code=base,  -- need to fill later
max_max_epoch = 22,
max_epoch = 4,
tolerance = 0.0, -- 0.1
minlr = 0.01,
max_grad_norm=5.0,}

local function transfer_data(x) return x:cuda() end
local state_train, state_valid, state_test
local train, valid, test
local models = {}
local idx2word, idx2cnt 
local train_vec, valid_vec, test_vec, check_conflict, vec_mapping
local mapx = {} local mapy = {}
local poses = {} start_poses = {}

local function lstm(x, x_r, prev_c, prev_h )
    -- Calculate all four gates in one go
    local i2hnn = nn.Linear(params.rnn_size, 4 * params.rnn_size)
    local h2hnn = nn.Linear(params.rnn_size, 4 * params.rnn_size)
    local i2h = i2hnn(x)
    local h2h = h2hnn(prev_h)
    local gates = nn.CAddTable()({i2h, h2h})

    -- Reshape to (batch_size, n_gates, hid_size)
    -- Then slize the n_gates dimension, i.e dimension 2
    local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
    local sliced_gates = nn.SplitTable(2)(reshaped_gates)

    -- Use select gate to fetch each gate and apply nonlinearity
    local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
    local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
    local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
    local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

    local next_c           = nn.CAddTable()({nn.CMulTable()({forget_gate, prev_c}), nn.CMulTable()({in_gate, in_transform})})
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

-- new _r
    local i2h_r = i2hnn(x_r)
    local h2h_r = h2hnn(next_h)
    local gates_r = nn.CAddTable()({i2h_r, h2h_r})

    local reshaped_gates_r = nn.Reshape(4, params.rnn_size)(gates_r)
    local sliced_gates_r   = nn.SplitTable(2)(reshaped_gates_r)

    local in_gate_r        = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates_r))
    local in_transform_r   = nn.Tanh()(nn.SelectTable(2)(sliced_gates_r))
    local forget_gate_r    = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates_r))
    local out_gate_r       = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates_r))

    local next_c_r         = nn.CAddTable()({nn.CMulTable()({forget_gate_r, next_c}), nn.CMulTable()({in_gate_r, in_transform_r})})
    local next_h_r         = nn.CMulTable()({out_gate_r, nn.Tanh()(next_c_r)})

    return next_c, next_h, next_c_r, next_h_r
end

local function create_network()
    local x                = nn.Identity()()
    local x_r              = nn.Identity()()
    local y                = nn.Identity()()
    local y_r              = nn.Identity()()
    local prev_s           = nn.Identity()()
    local lt1               = LookupTable(params.vocab_code, params.rnn_size)
    local lt2               = LookupTable(params.vocab_code, params.rnn_size)
    local i                = {[1] = lt1(x), [2] = lt2(x_r)}
    local next_s           = {}
    local split            = {prev_s:split(2 * params.layers)}
    for layer_idx = 1, params.layers do
        local prev_c         = split[2 * layer_idx - 1]
        local prev_h         = split[2 * layer_idx ]
        local dropped        = nn.Dropout(params.dropout)(i[2 * layer_idx - 1])
        local dropped_r      = nn.Dropout(params.dropout)(i[2 * layer_idx])
        local next_c, next_h, next_c_r, next_h_r = lstm(dropped, dropped_r, prev_c, prev_h)
        table.insert(next_s, next_c_r)
        table.insert(next_s, next_h_r)
        i[2 * layer_idx + 1] = next_h
        i[2 * layer_idx + 2] = next_h_r
    end
    local h2y              = nn.Linear(params.rnn_size, params.vocab_code)
    local h2y_r            = nn.Linear(params.rnn_size, params.vocab_code)
    local dropped          = nn.Dropout(params.dropout)(i[2 * params.layers + 1]) -- top next_h
    local dropped_r        = nn.Dropout(params.dropout)(i[2 * params.layers + 2]) -- top next_h_r
    local pred             = nn.LogSoftMax()(h2y(dropped))
    local pred_r           = nn.LogSoftMax()(h2y_r(dropped_r))
    local err              = nn.ClassNLLCriterion()({pred, y})
    local err_r            = nn.ClassNLLCriterion()({pred_r, y_r})
    local module           = nn.gModule({x, y, prev_s, x_r, y_r}, {err, err_r, nn.Identity()(next_s)})
    module:getParameters():uniform(-params.init_weight, params.init_weight)
    return transfer_data(module)
end

local function setup(model, str, g, step_add) -- we need to know which model, and GPU g
    print("Creating a RNN LSTM network. "..str.." in GPU "..g)
    print(params)
    g_set_gpu(g)

    local core_network = create_network()
    model.paramx, model.paramdx = core_network:getParameters()
    model.s = {}
    model.ds = {}
    model.start_s = {}
    for j = 0, params.seq_length do
        model.s[j] = {}
        for d = 1, 2 * params.layers do
            model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
        end
    end
    for d = 1, 2 * params.layers do
        model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
        model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
    model.core_network = core_network
    model.rnns = g_cloneManyTimes(core_network, params.seq_length)
    model.norm_dw = 0
    model.err = transfer_data(torch.zeros(params.seq_length))
    model.err_r = transfer_data(torch.zeros(params.seq_length))
    poses[g] = 1 + (g - 1) * step_add
    start_poses[g] = 1 + (g - 1) * step_add
    --model.pos   = 1 + (g - 1) * step_add
    --model.start_pos = 1 + (g - 1) * step_add
end

local function reset_state(model, g)
    g_set_gpu(g)
    if model ~= nil and model.start_s ~= nil then 
        for d = 1, 2 * params.layers do model.start_s[d]:zero() end end
end
local function reset_ds(model)
    for d = 1, #model.ds do model.ds[d]:zero() end
end

local function fp(models, state) -- !! make sure label responds to model, fill the right label before using it
    for m = 1, #models do -- all model go
        g_set_gpu(m) -- pay attention to set gpu for :cuda() to move
        local model = models[m]
        if poses[m] + params.seq_length > state.data:size(1) then
            reset_state(model, m) -- poses[m] = 1 -- for recursive
            poses[m] = 1
        end
        g_replace_table(model.s[0], model.start_s)
        for i = 1, params.seq_length do
            local x =   torch.floor(state.data[poses[m]] / jinzhi):cuda()
            local x_r = torch.mod(state.data[poses[m]]:int(), jinzhi):cuda()
            local y =   x_r
            local y_r = torch.floor(state.data[poses[m] + 1] / jinzhi):cuda()
            local s = model.s[i - 1]
            model.err[i], model.err_r[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s, x_r, y_r}))
            poses[m] = poses[m] + 1
        end
        g_replace_table(model.start_s, model.s[params.seq_length])
    end

    local err_sum = 0.0
    for m = 1, #models do
        g_set_gpu(m) -- pay attention to set gpu for :cuda() to move
        local model = models[m]
        err_sum = err_sum + model.err:mean() + model.err_r:mean()
    end
    return err_sum / GPUs 
end

local function bp(models, state)
    for m = 1, #models do
        g_set_gpu(m) -- pay attention to set gpu for :cuda() to move
        local model = models[m]
        local paramdx = model.paramdx
        local paramx  = model.paramx
        paramdx:zero() reset_ds(model)
        for i = params.seq_length, 1, -1 do
            poses[m] = poses[m] - 1
            local x = torch.floor(state.data[poses[m]] / jinzhi):cuda()
            local x_r = torch.mod(state.data[poses[m]]:int(), jinzhi):cuda()
            local y =   x_r
            local y_r = torch.floor(state.data[poses[m] + 1] / jinzhi):cuda()
            local s = model.s[i - 1]
            local derr = transfer_data(torch.ones(1))
            local derr_r = transfer_data(torch.ones(1))
            local tmp = model.rnns[i]:backward({x, y, s, x_r, y_r}, {derr, derr_r, model.ds})[3]
            g_replace_table(model.ds, tmp)
            --cutorch.synchronize()
        end
        poses[m] = poses[m] + params.seq_length
    end
    cutorch.synchronizeAll()

    -- todo !!!! every minibatch*seq_length to update
    g_set_gpu(1)
    local sumparamdx = models[1].paramdx
    local tmpparamdx = torch.CudaTensor(sumparamdx:size())
    for m = 2, #models do
        local model = models[m]
        local paramdx = model.paramdx
        tmpparamdx:copy(paramdx)
        sumparamdx:add(tmpparamdx)
        --[[
        model.norm_dw = paramdx:norm()
        if model.norm_dw > params.max_grad_norm then
            local shrink_factor = params.max_grad_norm / model.norm_dw
            paramdx:mul(shrink_factor)
        end
        paramx:add(paramdx:mul(-params.lr))
        ]]
    end

    sumparamdx:mul(1.0/GPUs)
    models[1].norm_dw = sumparamdx:norm()
    if models[1].norm_dw > params.max_grad_norm then
        local shrink_factor = params.max_grad_norm / models[1].norm_dw
        sumparamdx:mul(shrink_factor)
    end

    local model_1 = models[1]
    local paramx_1 = model_1.paramx
    paramx_1:add(sumparamdx:mul(-params.lr))
    -- spread all to GPUs
    for m = 1, #models do
        local model = models[m]
        g_set_gpu(m)
        local paramx = model.paramx
        paramx:copy(paramx_1)
    end

    cutorch.synchronizeAll()
end

local function run_valid(model, state) -- models[1]
    print("run valid...")
    --g_set_gpu(1)
    g_replace_table(model.ds, model.start_s) -- for store start_s
    g_disable_dropout(model.rnns)
    local len = (state.data:size(1) - 1) / (params.seq_length)
    local perp = 0
    reset_state(model, 1)
    state.pos = 1
    for j = 1, len do
        g_replace_table(model.s[0], model.start_s)
        for i = 1, params.seq_length do
            local x =   torch.floor(state.data[state.pos] / jinzhi):cuda()
            local x_r = torch.mod(state.data[state.pos]:int(), jinzhi):cuda()
            local y =   x_r
            local y_r = torch.floor(state.data[state.pos + 1] / jinzhi):cuda()
            local s = model.s[i - 1]
            model.err[i], model.err_r[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s, x_r, y_r}))
            state.pos = state.pos + 1
        end
        g_replace_table(model.start_s, model.s[params.seq_length])
        perp = perp + model.err:mean() + model.err_r:mean()
    end
    print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
    g_enable_dropout(model.rnns)

    g_replace_table(model.start_s, model.ds) -- save back to model.start_s
    cutorch.synchronize()
    return torch.exp(perp / len)
end

local function run_test(model) -- models[1]
    --g_set_gpu(1)
    g_replace_table(model.ds, model.start_s) -- for store start_s
    g_disable_dropout(model.rnns)
    local perp = 0
    local len = state_test.data:size(1)

    reset_state(model, 1)
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1), 1 do
        local x = torch.floor(state_test.data[i] / jinzhi):cuda()
        local x_r = torch.mod(state_test.data[i]:int(), jinzhi):cuda()
        local y =   x_r
        local y_r = torch.floor(state_test.data[i + 1] / jinzhi):cuda()
        perp_tmp, perp_tmp_r, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0], x_r, y_r}))
        perp = perp + perp_tmp[1] + perp_tmp_r[1]
        g_replace_table(model.s[0], model.s[1])
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
    g_enable_dropout(model.rnns)
    g_replace_table(model.start_s, model.ds) -- save back to model.start_s
    cutorch.synchronize()
    return g_f3(torch.exp(perp / (len - 1)))
end



-- init part, for many kinds
local function traininit(model, str, g, step_add)
    print("train init ...")
    setup(model, str, g, step_add)
end
local function datastateinit()
    print("data state init")
    for m = 1, #models do
        local model = models[m]
        reset_state(model, m) poses[m] = start_poses[m] 
    end
end
local function data_prepare(mapx, mapy)
    print("data prepare")
    check_conflict(mapx, mapy, params.vocab_size, params.vocab_code)
    state_train.data = (ptb.traindataset2batch(vec_mapping(state_train.vec, mapx, mapy, jinzhi), params.batch_size))
    state_valid.data = (ptb.validdataset2batch(vec_mapping(state_valid.vec, mapx, mapy, jinzhi), params.batch_size))
    state_test.data =  (ptb.testdataset2batch(vec_mapping(state_test.vec, mapx, mapy,   jinzhi), params.batch_size))
end


local function training(models, round)
    -- if loading from file, these must be added
    params.lr  = params.start_lr-- for new start, adjust lr -> 1.0
    collectgarbage()

    local total_cases = 0
    local beginning_time = torch.tic()
    local start_time = torch.tic()
    print("Starting training.")
    local words_per_step = params.seq_length * params.batch_size * GPUs
    local epoch_size = torch.floor(state_train.data:size(1) / params.seq_length / GPUs)
    local perps
    local last_valid_perp = 11111111111111111111111111111
    local last_train_perp = 11111111111111111111111111111
    local last_checkpoint = {} local checkpoint = {}
    local last_delete_file
    local step = 0 local epoch = 0


    while epoch < params.max_max_epoch do
        local perp = fp(models, state_train)
        if perps == nil then perps = torch.zeros(epoch_size):add(perp) end
        perps[step % epoch_size + 1] = perp
        step = step + 1
        bp(models, state_train)
        total_cases = total_cases + words_per_step
        epoch = step / epoch_size
        if step % torch.round(epoch_size / 10) == 10 then
            local wps = torch.floor(total_cases / torch.toc(start_time))
            local since_beginning = g_d(torch.toc(beginning_time) / 60)
            -- lr decay by train perp
            local new_train_perp = torch.exp(perps:mean())
            if new_train_perp > last_train_perp then 
                params.lr = params.lr / params.decay
            end
            last_train_perp = new_train_perp

            print('epoch = ' .. g_f3(epoch) ..
            ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
            ', wps = ' .. wps ..
            ', dw:norm() = ' .. g_f3(models[1].norm_dw) ..
            ', lr = ' ..  g_f3(params.lr) ..
            ', since beginning = ' .. since_beginning .. ' mins.')
        end
        if step % epoch_size == 0 then
            -- first let's see one epoch change
            local new_valid_perp = run_valid(models[1], state_valid)
            if new_valid_perp > last_valid_perp + params.tolerance or epoch > params.max_epoch then  -- or params.lr < params.minlr  then --or step / epoch_size > 1.0 then -- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! for fast check, later delete
                params.lr = params.lr / params.decay
            end
            last_valid_perp = new_valid_perp

            checkpoint.model    = models[1] 
            checkpoint.params   = params 
            checkpoint.mapx     = mapx
            checkpoint.mapy     = mapy
            checkpoint.savefile = string.format('implus_round%d_epoch%.2f_%.2f', round, epoch, new_valid_perp)
            last_checkpoint = checkpoint

            -- for save every epoch
            print("running test...")
            --local test_perp = run_test(models[1])
            local test_perp = last_valid_perp -- final to run test
            checkpoint.savefile = "./models/" .. checkpoint.savefile ..".test"..test_perp.. ".model.t7"
            torch.save(checkpoint.savefile, checkpoint)
            print("saved ".. checkpoint.savefile)
            if last_delete_file ~= nil then
                print("delete last file: ".. last_delete_file.." for save storage!") io.popen("rm "..last_delete_file)
            end
            last_delete_file = checkpoint.savefile

            if params.lr < params.minlr then break end

        end
        
        if step % 33 == 0 then cutorch.synchronizeAll() collectgarbage() end
    end
    print("running test ...")
end


local function updatebest_c(model, info, round, state) -- models[1]
    print('update best c begin!!', info)
    local probx = torch.zeros(params.vocab_size, params.vocab_code)
    local proby = torch.zeros(params.vocab_size, params.vocab_code)

    g_set_gpu(1)
    reset_state(model, 1)
    state.pos = 1
    g_disable_dropout(model.rnns)
    local len = (state.data:size(1) - 1) / (params.seq_length)
    local tot = 0
    local start_time = torch.tic()
    print("training len = ", len)
    for j = 1, len do
    --for j = 1, 1 do
        g_replace_table(model.s[0], model.start_s)
        for i = 1, params.seq_length do
            local x = torch.floor(state.data[state.pos] / jinzhi):cuda()
            local x_r = torch.mod(state.data[state.pos]:int(), jinzhi):cuda()
            local y = x_r
            local y_r = torch.floor(state.data[state.pos + 1] / jinzhi):cuda()
            local s = model.s[i - 1]
            model.err[i], model.err_r[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s, x_r, y_r}))
            local lsm = model.rnns[i]:findModules('nn.LogSoftMax')
            assert(#lsm == 2)
            local loy = lsm[1].output
            local lox = lsm[2].output
            lox = torch.DoubleTensor(lox:size()):copy(lox)
            loy = torch.DoubleTensor(loy:size()):copy(loy)

            local ny = state.original[state.pos]
            for k = 1, ny:size(1) do
                local idx = ny[k] proby[idx] = proby[idx] - loy[k]
            end

            local nx = state.original[state.pos + 1]
            for k = 1, nx:size(1) do
                local idx = nx[k] probx[idx] = probx[idx] - lox[k]
            end
            state.pos = state.pos + 1
        end
        g_replace_table(model.start_s, model.s[params.seq_length])
        if j%torch.floor(len/20) == 10 then print("j = "..j.." time use = "..g_d(torch.toc(start_time) / 60).." mins") end
    end
    g_enable_dropout(model.rnns)

    local since_start = g_d(torch.toc(start_time) / 60)
    print("go through train ", since_start, "mins")
    print('output values of two tables probx and proby into files ./models/')

    local xfilename = './models/round'..round..'.probx.t7'
    local filex = io.open(xfilename, 'w')
    for word  = 1, params.vocab_size do
        for i = 1, params.vocab_code do
            filex:write(' '..probx[word][i])
        end
        filex:write('\n')
    end
    filex:close()

    local yfilename = './models/round'..round..'.proby.t7'
    local filey = io.open(yfilename, 'w')
    for word  = 1, params.vocab_size do
        for i = 1, params.vocab_code do
            filey:write(' '..proby[word][i])
        end
        filey:write('\n')
    end
    filey:close()

    --local xfilename = './models/round'..round..'.probx.t7'
    --local yfilename = './models/round'..round..'.proby.t7'

    print("round "..round.." file probx, proby saved! call c++ program to deal")
    local mapxyfilename = './models/round'..round..'.mapxy.t7'
    local cmd = "./implus "..xfilename..' '..yfilename..' '..mapxyfilename..' '..data_path..'/idx2word.txt'
    print(cmd)
    local handle = io.popen(cmd)
    handle:close()

    print('assign mapx, mapy')
    local mapxyfilename = './models/round'..round..'.mapxy.t7'
    local filemapxy = io.open(mapxyfilename, 'r')
    local idx = 1 local idy
    mapx = {}
    mapy = {}
    while true do
        local str = filemapxy:read()
        if str == nil then break end
        str = stringx.split(str)
        for idy = 1, params.vocab_code do
            local word = tonumber(str[idy])
            if word > 0 then
                mapx[word] = idx
                mapy[word] = idy
            end
        end
        idx = idx + 1
    end
    check_conflict(mapx, mapy, params.vocab_size, params.vocab_code)
end

local function randommapxy(vocab, base)
    mapx = {} mapy = {}
    local ind = {}
    for i = 1, base do
        for j = 1, base do
            table.insert(ind, {i, j})
        end
    end
    local r = torch.randperm(base * base)
    for w = 1, vocab do
        local p = ind[r[w]]
        mapx[w] = p[1]
        mapy[w] = p[2]
    end
end


local function gao()
    -- g_init_gpu({}) -- gpu is nil, then we use the first gpu by default
    if arg[1] == nil or arg[2] == nil then print("Useage: th main_c.lua 4(GPUs) ../../dataset/big-data/xx/ 200(setting base to 200, if no, sqrt(vocab_size) will be default) 2>&1 | tee log.txt") return end
    GPUs      = tonumber(arg[1])
    print("use ---- "..GPUs.." gpus !!!!!")
    data_path = arg[2]

    vocab = 0
    for line in io.lines(data_path.."/idx2word.txt") do
        vocab = vocab + 1
    end
    print("vacab = "..vocab)
    local sqrt = math.ceil(math.sqrt(vocab))
    if arg[3] ~= nil then
        base  = tonumber(arg[3])
        if base < sqrt or base > jinzhi then
            print("you setting base is too small, please set at least ", sqrt, " at most ", jinzhi)
            return
        end
    else
        base  = sqrt
    end
    print("base = "..base)
    params.vocab_size = vocab
    params.vocab_code = base

    train_vec = ptb.traindataset(data_path)
    valid_vec = ptb.validdataset(data_path)
    test_vec  = ptb.testdataset(data_path)
    idx2word    = ptb.idx2word
    idx2cnt     = ptb.idx2cnt
    check_conflict = ptb.check_conflict
    vec_mapping = ptb.vec_mapping
    mapx        = ptb.mapx   mapy        = ptb.mapy -- it's from sortmapxy
    -- we set a random one -- print("we use random map to init mapx, mapy") randommapxy(vocab, base)
    check_conflict(mapx, mapy, vocab, base)

    state_train = {original = ptb.traindataset2batch(train_vec, params.batch_size), vec = train_vec, }
    state_valid = {original = ptb.validdataset2batch(valid_vec, params.batch_size), vec = valid_vec, }
    state_test  = {original = ptb.testdataset2batch(test_vec,   params.batch_size), vec = test_vec,  }
    if not path.exists('./models/') then lfs.mkdir('./models/') end

    local step_add = torch.floor((state_train.original:size(1) - 1) / GPUs)

    for g = 1, GPUs do
        local model = {}
        traininit(model, 'model init, using gpu '..g, g, step_add)
        table.insert(models, model)
    end

    for round = 0, 10000 do -- 0 is first init mapx, mapy to run
        print("------------------------------- round "..round.." begin!! ---------------------------------")
        if round == 0 then
            params.decay = 1.2
            params.max_epoch     = 10
            params.max_max_epoch = 44
        else 
            params.decay = 1.5
            params.max_epoch     = 1
            params.max_max_epoch = 10
        end
        if round > 0 then
            updatebest_c(models[1], "new technique!", round, state_train)
        end

        data_prepare(mapx, mapy)
        datastateinit()
        training(models, round)
    end
end

gao()
