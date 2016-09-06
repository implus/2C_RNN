local stringx = require('pl.stringx')
local vocab_idx = 0
local mapx = {} local mapy = {}
local word2idx =  {}
local idx2word  = {}
local idx2cnt   = {}

-- Stacks replicated, shifted versions of x_inp
-- into a single matrix of size x_inp:size(1) x batch_size.
local function replicate(x_inp, batch_size)
    local s = x_inp:size(1)
    local x = torch.zeros(torch.floor(s / batch_size), batch_size)
    local finish = 0 local start
    for i = 1, batch_size do
        start = finish + 1
        finish = start + x:size(1) - 1
        x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish))
    end
    return x
end

local function check_conflict(mapx0, mapy0, vocab, base)
    print('check if conflict with mapx, mapy:...')
    allset = {} distinct = {}
    for i = 1, vocab do
        if mapx0[i] <= 0 or mapx0[i] > base then
            print('error! range out 0!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        elseif mapy0[i] <= 0 or mapy0[i] > base then
            print('error! range out 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        end

        newval = base * (mapx0[i] - 1) + mapy0[i] - 1 
        if allset[newval] == nil then table.insert(distinct, newval) end
        allset[newval] = 1
    end
    if #distinct ~= vocab then
        print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!! #alldistince = ", #distinct, " not match with vocab = ", vocab)
    end
end

local tds = require('tds')
local function load_data(fname, data_path)
    local data = tds.Vec()
    file = io.open(fname)
    while true do
        local str = file:read()
        if str == nil then break end
        str = stringx.split(str)
        for i = 1, #str do
            data:insert(tonumber(str[i]))
        end
    end
    file:close()

    print(string.format("Loading %s, size of data = %d", fname, #data))

    if #mapx == 0 or #mapy == 0 then
        local filename = string.format('%s/sortmapxy.txt', data_path)
        --local filename = './models/round4.mapxy.t7'
        print('load '..filename..' mapxy ')
        fmapxy = io.open(filename, 'r')
        local row = 1 
        while true do
            local str = fmapxy:read()
            if str == nil then break end
            str = stringx.split(str)
            for col = 1, #str do
                local w = tonumber(str[col])
                if w > 0 then 
                    mapx[w] = row
                    mapy[w] = col
                end
            end
            row = row + 1
        end
    end
<<<<<<< HEAD
=======
    if #idx2word == 0 then
        local f = string.format('%s/idx2word.txt', data_path)
        print('load '..f..' for vaocab')
        fr = io.open(f, 'r')
        local id = 0
        while true do
            local str = fr:read()
            if str == nil then break end
            id = id + 1
            if str == '<S>' then str = '<s>'
            elseif str == '</S>' then  str = '</s>'
            end
            idx2word[id] = str
        end
        print('word vocab = ', #idx2word)
    end
>>>>>>> 6668573e372101b2eb7257f0fba47e70c2724eaa


    local x = torch.zeros(#data)
    for i = 1, #data do
        local w = data[i]
        if i == 1 then print("first data is "..data[i]) end
        x[i] = w
    end
    collectgarbage()
    return x
end

local function traindataset2batch(v, batch_size)
    local x = v:clone()
    x = replicate(x, batch_size)
    return x
end
local function traindataset(data_path, id)
    local v = load_data(data_path .. "/train_"..id..".txt", data_path)
    return v
end

-- Intentionally we repeat dimensions without offseting.
-- Pass over this batch corresponds to the fully sequential processing.
local function testdataset2batch(v, batch_size)
    local x = v:clone()
    x = x:resize(x:size(1), 1):expand(x:size(1), batch_size)
    return x
end
local function testdataset(data_path, id)
    local v = load_data(data_path .. "/test_"..id..".txt", data_path)
    return v
end


local function validdataset2batch(v, batch_size)
    local x = v:clone()
    x = replicate(x, batch_size)
    return x
end
local function validdataset(data_path, id)
    local v = load_data(data_path .. "/valid_"..id..".txt", data_path)
    return v
end


local function vec_mapping(x, mapx, mapy, jinzhi)
    local x0 = x:clone() assert(x0:dim() == 1)
    for i = 1, x0:size(1) do
        x0[i] = mapx[x0[i]] * jinzhi + mapy[x0[i]]
    end
    return x0
end

return {
traindataset = traindataset,
testdataset  = testdataset,
validdataset = validdataset,
traindataset2batch = traindataset2batch,
testdataset2batch  = testdataset2batch,
validdataset2batch = validdataset2batch,
vec_mapping = vec_mapping,
idx2word    = idx2word,
idx2cnt     = idx2cnt,
check_conflict = check_conflict,
mapx        = mapx,
mapy        = mapy,
}
