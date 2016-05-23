local stringx = require('pl.stringx')
local file = require('pl.file')

local ptb_path = "./data/"

local vocab_idx = 0
local vocab_map = {}
local vocab_0   = {}
local vocab_1   = {}

-- Stacks replicated, shifted versions of x_inp
-- into a single matrix of size x_inp:size(1) x batch_size.
local function replicate(x_inp, batch_size)
    local s = x_inp:size(1)
    local x = torch.zeros(torch.floor(s / batch_size), batch_size)
    for i = 1, batch_size do
        local start = torch.round((i - 1) * s / batch_size) + 1
        local finish = start + x:size(1) - 1
        x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish))
    end
    return x
end


local function cmp(x, y)
    return x > y
end

local function load_data(fname)
    local data = file.read(fname)
    data = stringx.replace(data, '\n', '<eos>')
    data = stringx.split(data)
    print(string.format("Loading %s, size of data = %d", fname, #data))
    local x = torch.zeros(#data)
    distinct_data = {}
    for i = 1, #data do
        if vocab_map[data[i]] == nil then
            vocab_idx = vocab_idx + 1
            vocab_map[data[i]] = vocab_idx
            table.insert(distinct_data, data[i])
        end
        x[i] = vocab_map[data[i]]
    end

    if #vocab_0 > 0 then
        return x
    end
    print("deal with vocab_0,_1, attention must be once")

    base = torch.sqrt(vocab_idx)

    if base * base ~= vocab_idx then
        base = base + 1
    end
    print("sqrt base = ", base)

--  get vocab_0 and vocab_1
    -- for vocab x dim --> vocab_0
    table.sort(distinct_data)
    print("distinct words = ", #distinct_data)
    print('vocab_idx      = ', vocab_idx)
    for i, w in pairs(distinct_data) do
	assert(vocab_0[vocab_map[w]] == nil)
 	assert(i > 0)
        vocab_0[vocab_map[w]] = torch.floor((i - 1) / base) + 1
        vocab_1[vocab_map[w]] = torch.floor((i - 1) % base) + 1
    end
    
    print('max = ', table.maxn(vocab_0), 'check right:...' )
    allset = {}
    for i = 1, vocab_idx do
        if vocab_0[i] <= 0 or vocab_0[i] > base then
            print('range out 0!')
        elseif vocab_1[i] <= 0 or vocab_1[i] > base then
            print('range out 1!')
        end

        newval = base * (vocab_0[i] - 1) + vocab_1[i] -- pay attention 0 is not counted in size
        allset[newval] = 1
    end
    if #allset ~= vocab_idx then
        print("error!!!! #allset = ", #allset)
    end

    -- print some
    for i = 1, 10 do
        print("data[", i, "] = ", data[i], "\t\t(", vocab_0[vocab_map[data[i]]], ",", vocab_1[vocab_map[data[i]]], ")")
    end

    return x
end

local function traindataset(batch_size)
    local x = load_data(ptb_path .. "ptb.train.txt")
    x = replicate(x, batch_size)
    return x
end

-- Intentionally we repeat dimensions without offseting.
-- Pass over this batch corresponds to the fully sequential processing.
local function testdataset(batch_size)
    local x = load_data(ptb_path .. "ptb.test.txt")
    x = x:resize(x:size(1), 1):expand(x:size(1), batch_size)
    return x
end

local function validdataset(batch_size)
    local x = load_data(ptb_path .. "ptb.valid.txt")
    x = replicate(x, batch_size)
    return x
end

local function traindataset_partition(batch_size)
    local x = traindataset(batch_size)
    local res = {}
    local x0 = x:clone()
    local s0 = x0:storage()
    for i = 1, s0:size() do
        s0[i] = vocab_0[s0[i]]
    end
    table.insert(res, x0)
    
    local x1 = x:clone()
    local s1 = x1:storage()
    for i = 1, s1:size() do
        s1[i] = vocab_1[s1[i]]
    end
    table.insert(res, x1)
    return res
end

local function testdataset_partition(batch_size)
    local x = testdataset(batch_size)
    local res = {}
    local x0 = x:clone()
    local s0 = x0:storage()
    for i = 1, s0:size() do
        s0[i] = vocab_0[s0[i]]
    end
    table.insert(res, x0)
    
    local x1 = x:clone()
    local s1 = x1:storage()
    for i = 1, s1:size() do
        s1[i] = vocab_1[s1[i]]
    end
    table.insert(res, x1)
    return res
end

local function validdataset_partition(batch_size)
    local x = validdataset(batch_size)
    local res = {}
    local x0 = x:clone()
    local s0 = x0:storage()
    for i = 1, s0:size() do
        s0[i] = vocab_0[s0[i]]
    end
    table.insert(res, x0)
    
    local x1 = x:clone()
    local s1 = x1:storage()
    for i = 1, s1:size() do
        s1[i] = vocab_1[s1[i]]
    end
    table.insert(res, x1)
    return res
end

return {traindataset=traindataset,
testdataset=testdataset,
validdataset=validdataset,
traindataset_partition = traindataset_partition,
validdataset_partition = validdataset_partition,
testdataset_partition  = testdataset_partition
}
