// GPU Radix Sort - Safe O(n) Implementation
// This implementation uses a two-phase prefix sum to avoid inter-workgroup deadlocks
// that can occur on Apple M1 and similar architectures.

// Constants are prepended before pipeline creation:
// const histogram_sg_size
// const histogram_wg_size
// const rs_radix_log2
// const rs_radix_size
// const rs_keyval_size
// const rs_histogram_block_rows
// const rs_scatter_block_rows

struct GeneralInfo {
    num_keys: u32,
    padded_size: u32,
    even_pass: u32,
    odd_pass: u32,
};

@group(0) @binding(0)
var<storage, read_write> infos: GeneralInfo;
@group(0) @binding(1)
var<storage, read_write> histograms: array<atomic<u32>>;
@group(0) @binding(2)
var<storage, read_write> keys: array<u32>;
@group(0) @binding(3)
var<storage, read_write> keys_b: array<u32>;
@group(0) @binding(4)
var<storage, read_write> payload_a: array<u32>;
@group(0) @binding(5)
var<storage, read_write> payload_b: array<u32>;

// Memory layout of histograms buffer:
// +-----------------------------------------+ <-- 0
// | global_histograms[keyval_size][radix]   |  (global prefix sums per pass)
// +-----------------------------------------+ <-- keyval_size * radix_size
// | block_histograms[num_blocks][radix]     |  (per-block local counts, reused per pass)
// +-----------------------------------------+ <-- keyval_size * radix_size + num_blocks * radix_size
// | block_offsets[num_blocks][radix]        |  (per-block global offsets)
// +-----------------------------------------+

// Workgroup shared memory
var<workgroup> smem: array<atomic<u32>, rs_radix_size>;
var<private> kv: array<u32, rs_histogram_block_rows>;
var<private> pv: array<u32, rs_histogram_block_rows>;
var<private> kr: array<u32, rs_scatter_block_rows>;

// --------------------------------------------------------------------------------------------------------------
// Helper functions
// --------------------------------------------------------------------------------------------------------------

fn get_scatter_block_kvs() -> u32 {
    return histogram_wg_size * rs_scatter_block_rows;
}

fn get_num_scatter_blocks(num_keys: u32) -> u32 {
    let scatter_block_kvs = get_scatter_block_kvs();
    return (num_keys + scatter_block_kvs - 1u) / scatter_block_kvs;
}

// Offset to block histograms (after global histograms)
fn block_histograms_offset() -> u32 {
    return rs_keyval_size * rs_radix_size;
}

// Offset to block offsets (after block histograms)
fn block_offsets_offset(num_blocks: u32) -> u32 {
    return block_histograms_offset() + num_blocks * rs_radix_size;
}

fn zero_smem(lid: u32) {
    if lid < rs_radix_size {
        atomicStore(&smem[lid], 0u);
    }
}

fn histogram_load(digit: u32) -> u32 {
    return atomicLoad(&smem[digit]);
}

fn histogram_store(digit: u32, count: u32) {
    atomicStore(&smem[digit], count);
}

// --------------------------------------------------------------------------------------------------------------
// Phase 0: Zero histograms and pad keys
// --------------------------------------------------------------------------------------------------------------
@compute @workgroup_size({histogram_wg_size})
fn zero_histograms(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    if gid.x == 0u {
        infos.even_pass = 0u;
        infos.odd_pass = 1u;
    }
    
    let scatter_block_kvs = get_scatter_block_kvs();
    let num_blocks = get_num_scatter_blocks(infos.num_keys);
    
    // Total histogram memory to zero:
    // - Global histograms: keyval_size * radix_size
    // - Block histograms: num_blocks * radix_size  
    // - Block offsets: num_blocks * radix_size
    let total_histogram_size = rs_keyval_size * rs_radix_size + 2u * num_blocks * rs_radix_size;
    
    // Also need to pad keys
    let padding_needed = infos.padded_size - infos.num_keys;
    let total_work = total_histogram_size + padding_needed;
    
    let line_size = nwg.x * {histogram_wg_size}u;
    for (var cur_index = gid.x; cur_index < total_work; cur_index += line_size) {
        if cur_index < total_histogram_size {
            atomicStore(&histograms[cur_index], 0u);
        } else {
            let key_idx = infos.num_keys + cur_index - total_histogram_size;
            keys[key_idx] = 0xFFFFFFFFu;
        }
    }
}

// --------------------------------------------------------------------------------------------------------------
// Phase 1: Calculate per-block histograms for all passes
// Each workgroup computes histogram for its block and stores it
// --------------------------------------------------------------------------------------------------------------

fn fill_kv_from_keys(wid: u32, lid: u32) {
    let rs_block_keyvals = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = wid * rs_block_keyvals + lid;
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_wg_size;
        kv[i] = keys[pos];
    }
}

fn fill_kv_from_keys_b(wid: u32, lid: u32) {
    let rs_block_keyvals = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = wid * rs_block_keyvals + lid;
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_wg_size;
        kv[i] = keys_b[pos];
    }
}

fn calculate_histogram_pass(pass_: u32, lid: u32) {
    zero_smem(lid);
    workgroupBarrier();
    
    for (var j = 0u; j < rs_histogram_block_rows; j++) {
        let u_val = bitcast<u32>(kv[j]);
        let digit = extractBits(u_val, pass_ * rs_radix_log2, rs_radix_log2);
        atomicAdd(&smem[digit], 1u);
    }
    
    workgroupBarrier();
    
    // Add to global histogram for this pass
    let global_histogram_offset = rs_radix_size * pass_ + lid;
    if lid < rs_radix_size {
        let count = atomicLoad(&smem[lid]);
        if count > 0u {
            atomicAdd(&histograms[global_histogram_offset], count);
        }
    }
}

@compute @workgroup_size({histogram_wg_size})
fn calculate_histogram(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    fill_kv_from_keys(wid.x, lid.x);
    
    // Calculate histograms for all passes
    calculate_histogram_pass(3u, lid.x);
    calculate_histogram_pass(2u, lid.x);
    calculate_histogram_pass(1u, lid.x);
    calculate_histogram_pass(0u, lid.x);
}

// --------------------------------------------------------------------------------------------------------------
// Phase 2: Prefix sum over global histogram
// Converts global histogram counts to exclusive prefix sums
// --------------------------------------------------------------------------------------------------------------

fn prefix_reduce_smem(lid: u32) {
    // Up-sweep (reduce) phase
    var offset = 1u;
    for (var d = rs_radix_size >> 1u; d > 0u; d = d >> 1u) {
        workgroupBarrier();
        if lid < d {
            let ai = offset * (2u * lid + 1u) - 1u;
            let bi = offset * (2u * lid + 2u) - 1u;
            atomicAdd(&smem[bi], atomicLoad(&smem[ai]));
        }
        offset = offset << 1u;
    }
    
    // Clear last element
    if lid == 0u {
        atomicStore(&smem[rs_radix_size - 1u], 0u);
    }
    
    // Down-sweep phase
    for (var d = 1u; d < rs_radix_size; d = d << 1u) {
        offset = offset >> 1u;
        workgroupBarrier();
        if lid < d {
            let ai = offset * (2u * lid + 1u) - 1u;
            let bi = offset * (2u * lid + 2u) - 1u;
            
            let t = atomicLoad(&smem[ai]);
            atomicStore(&smem[ai], atomicLoad(&smem[bi]));
            atomicAdd(&smem[bi], t);
        }
    }
}

@compute @workgroup_size({prefix_wg_size})
fn prefix_histogram(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    // wid.x is the pass index (0-3), inverted so pass 3 is first in memory
    let histogram_base = (rs_keyval_size - 1u - wid.x) * rs_radix_size;
    let histogram_offset = histogram_base + lid.x;
    
    // Load histogram into shared memory
    atomicStore(&smem[lid.x], atomicLoad(&histograms[histogram_offset]));
    atomicStore(&smem[lid.x + {prefix_wg_size}u], atomicLoad(&histograms[histogram_offset + {prefix_wg_size}u]));
    
    prefix_reduce_smem(lid.x);
    workgroupBarrier();
    
    // Store back the exclusive prefix sum
    atomicStore(&histograms[histogram_offset], atomicLoad(&smem[lid.x]));
    atomicStore(&histograms[histogram_offset + {prefix_wg_size}u], atomicLoad(&smem[lid.x + {prefix_wg_size}u]));
}

// --------------------------------------------------------------------------------------------------------------
// Phase 3a: Calculate per-block histograms for current pass
// This is run before each scatter pass
// --------------------------------------------------------------------------------------------------------------

fn store_block_histogram(pass_: u32, wid: u32, lid: u32, num_blocks: u32) {
    zero_smem(lid);
    workgroupBarrier();
    
    for (var j = 0u; j < rs_histogram_block_rows; j++) {
        let u_val = bitcast<u32>(kv[j]);
        let digit = extractBits(u_val, pass_ * rs_radix_log2, rs_radix_log2);
        atomicAdd(&smem[digit], 1u);
    }
    
    workgroupBarrier();
    
    // Store block histogram
    if lid < rs_radix_size {
        let block_hist_idx = block_histograms_offset() + wid * rs_radix_size + lid;
        atomicStore(&histograms[block_hist_idx], atomicLoad(&smem[lid]));
    }
}

@compute @workgroup_size({histogram_wg_size})
fn calculate_block_histogram_even(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let cur_pass = infos.even_pass * 2u;
    let num_blocks = get_num_scatter_blocks(infos.num_keys);
    
    fill_kv_from_keys(wid.x, lid.x);
    store_block_histogram(cur_pass, wid.x, lid.x, num_blocks);
}

@compute @workgroup_size({histogram_wg_size})
fn calculate_block_histogram_odd(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let cur_pass = infos.odd_pass * 2u + 1u;
    let num_blocks = get_num_scatter_blocks(infos.num_keys);
    
    fill_kv_from_keys_b(wid.x, lid.x);
    store_block_histogram(cur_pass, wid.x, lid.x, num_blocks);
}

// --------------------------------------------------------------------------------------------------------------
// Phase 3b: Calculate block offsets using sequential prefix sum
// Single workgroup computes prefix sum across all blocks for current pass
// --------------------------------------------------------------------------------------------------------------

@compute @workgroup_size({histogram_wg_size})
fn calculate_block_offsets_even(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let cur_pass = infos.even_pass * 2u;
    let num_blocks = get_num_scatter_blocks(infos.num_keys);
    
    // Each thread handles one digit
    if lid.x >= rs_radix_size {
        return;
    }
    
    let digit = lid.x;
    
    // Get global prefix for this digit
    let global_prefix_idx = cur_pass * rs_radix_size + digit;
    var running_sum = atomicLoad(&histograms[global_prefix_idx]);
    
    // Sequential scan across all blocks for this digit
    for (var block = 0u; block < num_blocks; block++) {
        let block_hist_idx = block_histograms_offset() + block * rs_radix_size + digit;
        let block_offset_idx = block_offsets_offset(num_blocks) + block * rs_radix_size + digit;
        
        // Store the exclusive prefix (offset where this block's keys for this digit start)
        atomicStore(&histograms[block_offset_idx], running_sum);
        
        // Add this block's count to running sum
        running_sum += atomicLoad(&histograms[block_hist_idx]);
    }
}

@compute @workgroup_size({histogram_wg_size})
fn calculate_block_offsets_odd(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let cur_pass = infos.odd_pass * 2u + 1u;
    let num_blocks = get_num_scatter_blocks(infos.num_keys);
    
    if lid.x >= rs_radix_size {
        return;
    }
    
    let digit = lid.x;
    
    let global_prefix_idx = cur_pass * rs_radix_size + digit;
    var running_sum = atomicLoad(&histograms[global_prefix_idx]);
    
    for (var block = 0u; block < num_blocks; block++) {
        let block_hist_idx = block_histograms_offset() + block * rs_radix_size + digit;
        let block_offset_idx = block_offsets_offset(num_blocks) + block * rs_radix_size + digit;
        
        atomicStore(&histograms[block_offset_idx], running_sum);
        running_sum += atomicLoad(&histograms[block_hist_idx]);
    }
}

// --------------------------------------------------------------------------------------------------------------
// Phase 3c: Scatter keys using pre-computed offsets
// No inter-workgroup synchronization needed - offsets are already computed
// --------------------------------------------------------------------------------------------------------------

var<workgroup> scatter_smem: array<u32, rs_mem_dwords>;

fn fill_kv_pv_even(wid: u32, lid: u32) {
    let subgroup_id = lid / histogram_sg_size;
    let subgroup_invoc_id = lid - subgroup_id * histogram_sg_size;
    let subgroup_keyvals = rs_scatter_block_rows * histogram_sg_size;
    let rs_block_keyvals = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = wid * rs_block_keyvals + subgroup_id * subgroup_keyvals + subgroup_invoc_id;
    
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        kv[i] = keys[pos];
    }
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        pv[i] = payload_a[pos];
    }
}

fn fill_kv_pv_odd(wid: u32, lid: u32) {
    let subgroup_id = lid / histogram_sg_size;
    let subgroup_invoc_id = lid - subgroup_id * histogram_sg_size;
    let subgroup_keyvals = rs_scatter_block_rows * histogram_sg_size;
    let rs_block_keyvals = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = wid * rs_block_keyvals + subgroup_id * subgroup_keyvals + subgroup_invoc_id;
    
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        kv[i] = keys_b[pos];
    }
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        pv[i] = payload_b[pos];
    }
}

fn scatter_phase(pass_: u32, lid: vec3<u32>, wid: vec3<u32>, num_blocks: u32) {
    let subgroup_id = lid.x / histogram_sg_size;
    let subgroup_offset = subgroup_id * histogram_sg_size;
    let subgroup_tid = lid.x - subgroup_offset;
    let subgroup_count = {scatter_wg_size}u / histogram_sg_size;
    
    // Step 1: Calculate local ranks within workgroup using simulated match
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        let u_val = bitcast<u32>(kv[i]);
        let digit = extractBits(u_val, pass_ * rs_radix_log2, rs_radix_log2);
        
        // Broadcast digit to shared memory
        atomicStore(&smem[lid.x], digit);
        workgroupBarrier();
        
        var count = 0u;
        var rank = 0u;
        
        for (var j = 0u; j < histogram_sg_size; j++) {
            if atomicLoad(&smem[subgroup_offset + j]) == digit {
                count += 1u;
                if j <= subgroup_tid {
                    rank += 1u;
                }
            }
        }
        
        kr[i] = (count << 16u) | rank;
        workgroupBarrier();
    }
    
    // Step 2: Build workgroup-local histogram and compute local offsets
    zero_smem(lid.x);
    workgroupBarrier();
    
    for (var i = 0u; i < subgroup_count; i++) {
        if subgroup_id == i {
            for (var j = 0u; j < rs_scatter_block_rows; j++) {
                let v = bitcast<u32>(kv[j]);
                let digit = extractBits(v, pass_ * rs_radix_log2, rs_radix_log2);
                let prev = histogram_load(digit);
                let rank = kr[j] & 0xFFFFu;
                let count = kr[j] >> 16u;
                kr[j] = prev + rank;
                
                if rank == count {
                    histogram_store(digit, prev + count);
                }
            }
        }
        workgroupBarrier();
    }
    
    // Step 3: Load pre-computed block offsets into shared memory
    if lid.x < rs_radix_size {
        let block_offset_idx = block_offsets_offset(num_blocks) + wid.x * rs_radix_size + lid.x;
        scatter_smem[lid.x] = atomicLoad(&histograms[block_offset_idx]);
    }
    workgroupBarrier();
    
    // Step 4: Compute exclusive prefix of local histogram
    prefix_reduce_smem(lid.x);
    workgroupBarrier();
    
    // Step 5: Convert local rank to local index
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        let v = bitcast<u32>(kv[i]);
        let digit = extractBits(v, pass_ * rs_radix_log2, rs_radix_log2);
        let local_prefix = histogram_load(digit);
        let idx = local_prefix + kr[i];
        kr[i] |= (idx << 16u);
    }
    workgroupBarrier();
    
    // Step 6: Reorder keys within workgroup
    let smem_reorder_offset = rs_radix_size;
    let smem_base = smem_reorder_offset + lid.x;
    
    // Store keys to sorted location
    for (var j = 0u; j < rs_scatter_block_rows; j++) {
        let smem_idx = smem_reorder_offset + (kr[j] >> 16u) - 1u;
        scatter_smem[smem_idx] = bitcast<u32>(kv[j]);
    }
    workgroupBarrier();
    
    // Load keys from sorted location
    for (var j = 0u; j < rs_scatter_block_rows; j++) {
        kv[j] = scatter_smem[smem_base + j * {scatter_wg_size}u];
    }
    workgroupBarrier();
    
    // Store payload to sorted location
    for (var j = 0u; j < rs_scatter_block_rows; j++) {
        let smem_idx = smem_reorder_offset + (kr[j] >> 16u) - 1u;
        scatter_smem[smem_idx] = pv[j];
    }
    workgroupBarrier();
    
    // Load payload from sorted location
    for (var j = 0u; j < rs_scatter_block_rows; j++) {
        pv[j] = scatter_smem[smem_base + j * {scatter_wg_size}u];
    }
    workgroupBarrier();
    
    // Store rank to sorted location
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        let smem_idx = smem_reorder_offset + (kr[i] >> 16u) - 1u;
        scatter_smem[smem_idx] = kr[i];
    }
    workgroupBarrier();
    
    // Load rank from sorted location (only need lower 16 bits)
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        kr[i] = scatter_smem[smem_base + i * {scatter_wg_size}u] & 0xFFFFu;
    }
    workgroupBarrier();
    
    // Step 7: Convert local index to global index using pre-computed offsets
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        let v = bitcast<u32>(kv[i]);
        let digit = extractBits(v, pass_ * rs_radix_log2, rs_radix_log2);
        let block_offset = scatter_smem[digit];  // Pre-computed global offset for this block/digit
        kr[i] += block_offset - 1u;
    }
}

@compute @workgroup_size({scatter_wg_size})
fn scatter_even(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    if gid.x == 0u {
        infos.odd_pass = (infos.odd_pass + 1u) % 2u;
    }
    
    let cur_pass = infos.even_pass * 2u;
    let num_blocks = get_num_scatter_blocks(infos.num_keys);
    
    fill_kv_pv_even(wid.x, lid.x);
    scatter_phase(cur_pass, lid, wid, num_blocks);
    
    // Store to output buffer
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        keys_b[kr[i]] = kv[i];
    }
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        payload_b[kr[i]] = pv[i];
    }
}

@compute @workgroup_size({scatter_wg_size})
fn scatter_odd(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    if gid.x == 0u {
        infos.even_pass = (infos.even_pass + 1u) % 2u;
    }
    
    let cur_pass = infos.odd_pass * 2u + 1u;
    let num_blocks = get_num_scatter_blocks(infos.num_keys);
    
    fill_kv_pv_odd(wid.x, lid.x);
    scatter_phase(cur_pass, lid, wid, num_blocks);
    
    // Store to output buffer
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        keys[kr[i]] = kv[i];
    }
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        payload_a[kr[i]] = pv[i];
    }
}
