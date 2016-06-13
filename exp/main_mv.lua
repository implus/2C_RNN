--local data_path = "../../dataset/big-data/es/" --local vocab = 151989 --local base = 390
local INFO_SPEED = 1
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
local multiverso = require('multiverso')

local params = {
batch_size=400,
seq_length=20, -- back to one element seq_length
layers=1,
decay=1.2,
rnn_size=2048,
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
max_grad_norm=5.0,
}

local function transfer_data(x) return x:cuda() end
local state_train, state_valid, state_test
local train, valid, test
local model = {}
local idx2word, idx2cnt 
local train_vec, valid_vec, test_vec, check_conflict, vec_mapping
local mapx = {} local mapy = {}

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

local function setup(model, str) -- we need to know which model
    print("Creating a RNN LSTM network. "..str)
    print(params)
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
end

local function reset_state(model, state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then for d = 1, 2 * params.layers do model.start_s[d]:zero() end end
end
local function reset_ds(model)
    for d = 1, #model.ds do model.ds[d]:zero() end
end

local function fp(model, state) -- !! make sure label responds to model, fill the right label before using it
    g_replace_table(model.s[0], model.start_s)
    if state.pos + params.seq_length > state.data:size(1) then
        reset_state(model, state)
    end
    for i = 1, params.seq_length do
        local x = torch.floor(state.data[state.pos] / jinzhi)
        local x_r = torch.mod(state.data[state.pos]:int(), jinzhi):cuda()
        local y =   x_r
        local y_r = torch.floor(state.data[state.pos + 1] / jinzhi)
        local s = model.s[i - 1]
        model.err[i], model.err_r[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s, x_r, y_r}))
        state.pos = state.pos + 1
    end
    g_replace_table(model.start_s, model.s[params.seq_length])
    return model.err:mean() + model.err_r:mean()
end

local function bp(model, state)
    local paramdx = model.paramdx
    local paramx  = model.paramx
    paramdx:zero()
    reset_ds(model)
    for i = params.seq_length, 1, -1 do
        state.pos = state.pos - 1
        local x = torch.floor(state.data[state.pos] / jinzhi)
        local x_r = torch.mod(state.data[state.pos]:int(), jinzhi):cuda()
        local y =   x_r
        local y_r = torch.floor(state.data[state.pos + 1] / jinzhi)
        local s = model.s[i - 1]
        local derr = transfer_data(torch.ones(1))
        local derr_r = transfer_data(torch.ones(1))
        local tmp = model.rnns[i]:backward({x, y, s, x_r, y_r}, {derr, derr_r, model.ds})[3]
        g_replace_table(model.ds, tmp)
        --cutorch.synchronize()
    end
    state.pos = state.pos + params.seq_length

    model.norm_dw = paramdx:norm()
    if model.norm_dw > params.max_grad_norm then
        local shrink_factor = params.max_grad_norm / model.norm_dw
        paramdx:mul(shrink_factor)
    end
    tbh:add(paramdx:mul(-params.lr/ multiverso.num_workers))
    paramx:copy(tbh:get())

    infos:add({words_per_step, 0, 0, 0, 0})
    --paramx:add(paramdx:mul(-params.lr))
end

local function run_valid(model)
    local paramx  = model.paramx
    paramx:copy(tbh:get())

    reset_state(model, state_valid)
    g_disable_dropout(model.rnns)
    local len = (state_valid.data:size(1) - 1) / (params.seq_length)
    local perp = 0
    for i = 1, len do
        perp = perp + fp(model, state_valid)
    end
    print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
    g_enable_dropout(model.rnns)
    return torch.exp(perp / len)
end

local function run_test(model)
    local paramx  = model.paramx
    paramx:copy(tbh:get())

    reset_state(model, state_test)
    g_disable_dropout(model.rnns)
    local perp = 0
    local len = state_test.data:size(1)
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1), 1 do
        local x = torch.floor(state_test.data[i] / jinzhi)
        local x_r = torch.mod(state_test.data[i]:int(), jinzhi):cuda()
        local y =   x_r
        local y_r = torch.floor(state_test.data[i + 1] / jinzhi)
        perp_tmp, perp_tmp_r, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0], x_r, y_r}))
        perp = perp + perp_tmp[1] + perp_tmp_r[1]
        g_replace_table(model.s[0], model.s[1])
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
    g_enable_dropout(model.rnns)
    return g_f3(torch.exp(perp / (len - 1)))
end


-- init part, for many kinds
local function traininit(model, str)
    print("train init ...")
    setup(model, str)
end
local function datastateinit(step_gap)
    print("data state init")
    local states = {state_train, state_valid, state_test}
    for _, state in pairs(states) do reset_state(model, state) end
    print("worker: "..multiverso.worker_id.." state_train.pos = "..state_train.pos)
end
local function data_prepare(mapx, mapy)
    print("data prepare")
    check_conflict(mapx, mapy, params.vocab_size, params.vocab_code)
    state_train.data = transfer_data(ptb.traindataset2batch(vec_mapping(state_train.vec, mapx, mapy, jinzhi), params.batch_size))
    state_valid.data = transfer_data(ptb.validdataset2batch(vec_mapping(state_valid.vec, mapx, mapy, jinzhi), params.batch_size))
    state_test.data =  transfer_data(ptb.testdataset2batch(vec_mapping(state_test.vec, mapx, mapy,   jinzhi), params.batch_size))
end

local function training(model, round)
    -- if loading from file, these must be added
    local paramx = model.paramx
    local paramdx = model.paramdx
    params.lr  = params.start_lr-- for new start, adjust lr -> 1.0
    collectgarbage()

    local total_cases = 0
    local beginning_time = torch.tic()
    local start_time = torch.tic()
    print("Starting training.")
    words_per_step = params.seq_length * params.batch_size 
    if multiverso.is_master then
        infos:add(-infos:get())
        multiverso.barrier()
    else
        multiverso.barrier()
    end
    local epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
    local perps
    local last_valid_perp = 11111111111111111111111111111
    local last_train_perp = 11111111111111111111111111111
    local last_checkpoint = {} local checkpoint = {}
    local last_delete_file
    local step = 0 local epoch = 0

    while epoch < params.max_max_epoch do
        local perp = fp(model, state_train)
        if perps == nil then perps = torch.zeros(epoch_size):add(perp) end
        perps[step % epoch_size + 1] = perp
        step = step + 1
        bp(model, state_train)
        --total_cases = total_cases + words_per_step
        epoch = step / epoch_size
        if step % torch.round(epoch_size / 100) == 10 then
            --local wps = torch.floor(total_cases / torch.toc(start_time))
            local wps  = torch.floor(infos:get()[INFO_SPEED] / torch.toc(start_time))
            local since_beginning = g_d(torch.toc(beginning_time) / 60)
            -- lr decay by train perp
            local new_train_perp = torch.exp(perps:mean())
            if new_train_perp > last_train_perp then 
                params.lr = params.lr / params.decay
            end
            last_train_perp = new_train_perp

            if multiverso.is_master then
                print('epoch = ' .. g_f3(epoch) ..
                ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
                ', wps = ' .. wps ..
                ', dw:norm() = ' .. g_f3(model.norm_dw) ..
                ', lr = ' ..  g_f3(params.lr) ..
                ', since beginning = ' .. since_beginning .. ' mins.')
            end
        end
        if step % epoch_size == 0 then
            -- first let's see one epoch change
            local new_valid_perp = run_valid(model)
            if new_valid_perp > last_valid_perp + params.tolerance or epoch > params.max_epoch then  
                params.lr = params.lr / params.decay
            end
            last_valid_perp = new_valid_perp

            if multiverso.is_master then
                checkpoint.model    = model 
                checkpoint.params   = params 
                checkpoint.mapx     = mapx
                checkpoint.mapy     = mapy
                checkpoint.savefile = string.format('implus_round%d_epoch%.2f_%.2f', round, epoch, new_valid_perp)
                last_checkpoint = checkpoint

                checkpoint.savefile = './models/'..checkpoint.savefile..'.model.t7'
                torch.save(checkpoint.savefile, checkpoint)
                print("saving ".. checkpoint.savefile)
                if last_delete_file ~= nil then
                    print("delete last file: ".. last_delete_file.." for save storage!") io.popen("rm "..last_delete_file)
                end
                last_delete_file = checkpoint.savefile

                multiverso.barrier()
            else
                multiverso.barrier() -- one epoch, wait go together next
            end

            if params.lr < params.minlr then break end
        end
        
        --if step % 33 == 0 then cutorch.synchronize() collectgarbage() end
    end
    print("finish training round, go test..")

    if multiverso.is_master then
        -- run test
        print("running test..."..multiverso.worker_id)
        local test_perp = run_test(checkpoint.model)
        checkpoint.savefile = checkpoint.savefile ..".test"..test_perp.. ".model.t7"
        torch.save(checkpoint.savefile, checkpoint)
        print("saving ".. checkpoint.savefile)
        multiverso.barrier()
    else
        multiverso.barrier()
    end
end


local function updatebest_c(model, info, round, state)
    print('update best c begin!!', info)
    local probx = torch.zeros(params.vocab_size, params.vocab_code)
    local proby = torch.zeros(params.vocab_size, params.vocab_code)

    reset_state(model, state)
    g_disable_dropout(model.rnns)
    local len = (state.data:size(1) - 1) / (params.seq_length)
    local tot = 0
    local start_time = torch.tic()
    print("training len = ", len)
    for j = 1, len do
    --for j = 1, 1 do
        g_replace_table(model.s[0], model.start_s)
        for i = 1, params.seq_length do
            local x = torch.floor(state.data[state.pos] / jinzhi)
            local x_r = torch.mod(state.data[state.pos]:int(), jinzhi):cuda()
            local y = x_r
            local y_r = torch.floor(state.data[state.pos + 1] / jinzhi)
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
        if multiverso.is_master and j%torch.floor(len/20) == 10 then print("j = "..j.." time use = "..g_d(torch.toc(start_time) / 60).." mins") end
    end
    g_enable_dropout(model.rnns)
    reset_state(model, state)

    local since_start = g_d(torch.toc(start_time) / 60)
    print("go through train ", since_start, "mins")
    print('output values of two tables probx and proby into files ./models/')

    local xfilename = './models/round'..round..'.probx.'..multiverso.worker_id..'.t7'
    --torch.save(xfilename, probx)
        print('write into '..xfilename)
        local filex = io.open(xfilename, 'w')
        for word  = 1, params.vocab_size do
            for i = 1, params.vocab_code do
                filex:write(' '..probx[word][i])
            end
            filex:write('\n')
        end
        filex:close()
    print('saving '..xfilename)

    local yfilename = './models/round'..round..'.proby.'..multiverso.worker_id..'.t7'
    --torch.save(yfilename, proby)
        print('write into '..yfilename)
        local filey = io.open(yfilename, 'w')
        for word  = 1, params.vocab_size do
            for i = 1, params.vocab_code do
                filey:write(' '..proby[word][i])
            end
            filey:write('\n')
        end
        filey:close()
    print('saving '..yfilename)

    -- wait for 4 files to be done
    multiverso.barrier()
    --local xfilename = './models/round'..round..'.probx.t7'
    --local yfilename = './models/round'..round..'.proby.t7'
    if multiverso.is_master then
        -- merge 4 files into one first
        local probx = torch.zeros(params.vocab_size, params.vocab_code)
        local proby = torch.zeros(params.vocab_size, params.vocab_code)
        
        print('!!!!!!!!!!!! merge all loss matrix into one !!!!!!!!!!!!!!!!!!')
        local cmdx = './merge ./models/round'..round..'.probx.t7'
        local cmdy = './merge ./models/round'..round..'.proby.t7'
        for w = 0, multiverso.num_workers - 1 do
            local xfilename = './models/round'..round..'.probx.'..w..'.t7'
            local yfilename = './models/round'..round..'.proby.'..w..'.t7'
            cmdx = cmdx..' '..xfilename
            cmdy = cmdy..' '..yfilename
            print(cmdx)
            print(cmdy)
            local handle = io.popen(cmdx)
            handle:close()
            local handle = io.popen(cmdy)
            handle:close()
        end
        --[[
        for w = 0, multiverso.num_workers - 1 do
            local xfilename = './models/round'..round..'.probx.'..w..'.t7'
            local px = torch.load(xfilename)
            probx = probx + px
            local yfilename = './models/round'..round..'.proby.'..w..'.t7'
            local py = torch.load(yfilename)
            proby = proby + py
        end

        local xfilename = './models/round'..round..'.probx.t7'
        print('write into '..xfilename)
        local filex = io.open(xfilename, 'w')
        for word  = 1, params.vocab_size do
            for i = 1, params.vocab_code do
                filex:write(' '..probx[word][i])
            end
            filex:write('\n')
        end
        filex:close()

        local yfilename = './models/round'..round..'.proby.t7'
        print('write into '..yfilename)
        local filey = io.open(yfilename, 'w')
        for word  = 1, params.vocab_size do
            for i = 1, params.vocab_code do
                filey:write(' '..proby[word][i])
            end
            filey:write('\n')
        end
        filey:close()
        -- finish file 
        ]]
        local xfilename = './models/round'..round..'.probx.t7'
        local yfilename = './models/round'..round..'.proby.t7'

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

        delx = mapx_tbh:get()
        delmapx = torch.zeros(#mapx)
        for i = 1, #mapx do delmapx[i] = mapx[i] - delx[i] end
        mapx_tbh:add(delmapx)
        
        dely = mapy_tbh:get()
        delmapy = torch.zeros(#mapy)
        for i = 1, #mapy do delmapy[i] = mapy[i] - dely[i] end
        mapy_tbh:add(delmapy)

        multiverso.barrier()
    else
        multiverso.barrier()

        tmapx = mapx_tbh:get()
        for i = 1, #mapx do mapx[i] = tmapx[i] end
        tmapy = mapy_tbh:get()
        for i = 1, #mapy do mapy[i] = tmapy[i] end
        check_conflict(mapx, mapy, params.vocab_size, params.vocab_code)
    end
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
    --g_init_gpu({}) -- gpu is nil, then we use the first gpu by default
    multiverso.init()
    g_set_gpu(multiverso.worker_id() + 1)
    multiverso.num_workers = multiverso.num_workers()
    multiverso.worker_id = multiverso.worker_id()
    multiverso.is_master = (multiverso.worker_id == 0)

    if arg[1] == nil then print("useage: mpirun -np 4 th main_c.lua ../../dataset/big-data/xx/ 200(setting base to 200, if no, sqrt(vocab_size) will be default) 2>&1 | tee log.txt") return end
    data_path = arg[1]
    print('data_path:'..data_path)

    vocab = 0
    for line in io.lines(data_path.."/idx2word.txt") do
        vocab = vocab + 1
    end
    print("vacab = "..vocab)
    local sqrt = math.ceil(math.sqrt(vocab))
    if arg[2] ~= nil then
        base  = tonumber(arg[2])
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

    train_vec = ptb.traindataset(data_path, multiverso.worker_id + 1) -- 1, 2, 3, 4 (0 is all data)
    valid_vec = ptb.validdataset(data_path, multiverso.worker_id + 1)
    test_vec  = ptb.testdataset(data_path,  multiverso.worker_id + 1)
    idx2word    = ptb.idx2word
    idx2cnt     = ptb.idx2cnt
    check_conflict = ptb.check_conflict
    vec_mapping = ptb.vec_mapping
    mapx        = ptb.mapx   mapy        = ptb.mapy -- it's from sortmapxy
    -- we set a random one
    -- print("we use random map to init mapx, mapy") randommapxy(vocab, base)
    check_conflict(mapx, mapy, vocab, base)

    state_train = {original = ptb.traindataset2batch(train_vec, params.batch_size), vec = train_vec, }
    state_valid = {original = ptb.validdataset2batch(valid_vec, params.batch_size), vec = valid_vec, }
    state_test  = {original = ptb.testdataset2batch(test_vec,   params.batch_size), vec = test_vec,  }
    if not path.exists('./models/') then lfs.mkdir('./models/') end

    traininit(model, 'model init')
    tbh = multiverso.ArrayTableHandler:new(model.paramx:size(1))
    infos = multiverso.ArrayTableHandler:new(5)
    mapx_tbh = multiverso.ArrayTableHandler:new(#mapx)
    mapy_tbh = multiverso.ArrayTableHandler:new(#mapy)
    -- infos[0] = speed cnt;
    
    if multiverso.is_master then
        tbh:add(model.paramx)
        infos:add({0, 0, 0, 0, 0})
        multiverso.barrier()
    else
        multiverso.barrier()
        model.paramx:copy(tbh:get()) -- make sure all the initial values are the same for each worker
    end

    for round = 0, 10000 do -- 0 is first init mapx, mapy to run
        print("------------------------------- round "..round.." begin!! ---------------------------------")
        if round == 0 then
            params.decay = 1.15
            params.max_epoch     = 2 --10
            params.max_max_epoch = 10 --44
        else 
            params.decay = 1.2
            params.max_epoch     = 2 
            params.max_max_epoch = 6 --12
        end
        if round > 0 then
            model.paramx:copy(tbh:get())
            updatebest_c(model, "new technique!", round, state_train)
            multiverso.barrier()
        end

        data_prepare(mapx, mapy) -- prepare data into batch_size with mapx mapy in cpu
        datastateinit()          -- init from 0
        training(model, round)
    end

    multiverso.shutdown()
end

gao()
