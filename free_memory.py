import pynvml, torch, gc

pynvml.nvmlInit()
id = torch.cuda.current_device()
def mem_free():
    gc.collect()
    torch.cuda.empty_cache()
    handle = pynvml.nvmlDeviceGetHandleByIndex(id)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return int( info.free / 2**20 )

def mem_report(): print(f"free mem={mem_free()}")

def mem_allocate_mbs(n, fatal=False): 
    " allocate n MBs, return the var holding it on success, None on failure "
    if n < 6: return None # don't try to allocate less than 6MB
    try:
        d = int(2**9*n**0.5)
        return torch.ones((d, d)).cuda().contiguous()
    except Exception as e:
        if not fatal: return None
        raise e
        
def leave_free_mbs(n):
    " consume whatever memory is needed so that n MBs are left free "
    avail = mem_free()
    assert avail > n, f"already have less available mem than desired {n}MBs"
    consume = avail - n
    print(f"consuming {consume}MB to bring free mem to {n}MBs")
    return mem_allocate_mbs(consume, fatal=True)

def globals_unset(var_names):
    " this is useful for re-running the cell, so that it resets the initial state or cleanup at the end of the cell"
    for x in var_names: 
        if x in globals(): 
            del globals()[x]
            
def mem_get(n):
    print(f"have {mem_free():4d}, allocating {n}")
    return mem_allocate_mbs(n, fatal=True)

globals_unset(['x1', 'x2', 'x3', 'buf'])
_=torch.ones(1).cuda()# preload

# this ensures we always test the same thing
buf = leave_free_mbs(1600)
    
                   # legend: [free block]  {used block}
                   # [1600]
x1 = mem_get(512)  # {512}[1092]
x2 = mem_get(512)  # {512}{512}[576]
print(f"have {mem_free():4d}, reclaiming first 512")
del x1             # [512]{512}[576]
x3 = mem_get(1024) # shouldn't be able to allocate 1024 contiguous mem
print(f"have {mem_free():4d}")

# cleanup
globals_unset(['x1', 'x2', 'x3', 'buf'])

# this ensures we always test the same thing
buf = leave_free_mbs(1600)
    
                   # legend: [free block]  {used block}
                   # [1601]
x1 = mem_get(512)  # {512}[1089]
x2 = mem_get(512)  # {512}{512}[577]
print(f"have {mem_free():4d}, reclaiming first 512")
del x1             # [512]{512}[577]
x3 = mem_get(1024) # shouldn't be able to allocate 1024 contiguous mem
print(f"have {mem_free():4d}")
