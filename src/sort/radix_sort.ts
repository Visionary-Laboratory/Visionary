// GPU Radix Sort implementation - deadlock-free 3-phase scatter
// Replaces the original decoupled look-back (cross-workgroup spin-wait) with:
//   Phase 1 (scatter_local): each workgroup computes local histogram + writes reduction
//   Phase 2 (scatter_prefix_pass): single-workgroup pass scans all partition reductions
//   Phase 3 (scatter_apply): each workgroup reads precomputed prefix and scatters

import { radixSortShader } from '../shaders';
import { ISorter, SortedSplats } from './index';

// IMPORTANT: The following constants must be synced with the numbers in radix_sort.wgsl
export const HISTOGRAM_WG_SIZE = 256;
const RS_RADIX_LOG2 = 8; // 8-bit radices
const RS_RADIX_SIZE = 1 << RS_RADIX_LOG2; // 256 entries into the radix table
const RS_KEYVAL_SIZE = 32 / RS_RADIX_LOG2; // 4 passes for 32-bit keys
export const RS_HISTOGRAM_BLOCK_ROWS = 15;
const RS_SCATTER_BLOCK_ROWS = RS_HISTOGRAM_BLOCK_ROWS; // DO NOT CHANGE, shader assumes this
const PREFIX_WG_SIZE = 1 << 7; // 128, one thread operates on 2 prefixes at the same time
const SCATTER_WG_SIZE = 1 << 8; // 256

/**
 * Interface for the uniform buffer data.
 * The layout must match the `GeneralInfo` struct in the WGSL shader.
 */
interface GeneralInfo {
    keys_size: number;
    padded_size: number;
    passes: number;
    even_pass: number;
    odd_pass: number;
}

/**
 * Interface for the indirect dispatch buffer data.
 */
interface IndirectDispatch {
    dispatch_x: number;
    dispatch_y: number;
    dispatch_z: number;
}

/**
 * A container for all the GPU resources associated with sorting a particular point cloud.
 */
export interface PointCloudSortStuff extends SortedSplats {
    // Compatibility field for renderer - same as numPoints from SortedSplats
    num_points: number;
    // Uniform buffer holding general sorting info (key size, padding, etc.)
    sorter_uni: GPUBuffer;
    // Buffer for indirect dispatch commands
    sorter_dis: GPUBuffer;
    // Main bind group for the sorting compute passes
    sorter_bg: GPUBindGroup;
    // Bind group for rendering, with read-only access to sorted data
    sorter_render_bg: GPUBindGroup;
    // Bind group for the preprocessing step
    sorter_bg_pre: GPUBindGroup;
    // Internal memory buffer for histograms and partitions
    internal_mem: GPUBuffer;
    // Ping-pong buffers for keys
    key_a: GPUBuffer;
    key_b: GPUBuffer;
    // Ping-pong buffers for payloads (e.g., original indices)
    payload_a: GPUBuffer;
    payload_b: GPUBuffer;
    // Mapping from sorted slot -> source point index
    source_indices: GPUBuffer;
}

function shuffleArray(array: any[]) {
  const arr = array.slice();
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

export class GPURSSorter implements ISorter {
    private bindGroupLayout!: GPUBindGroupLayout;
    private renderBindGroupLayout!: GPUBindGroupLayout;
    private preprocessBindGroupLayout!: GPUBindGroupLayout;

    private zero_p!: GPUComputePipeline;
    private histogram_p!: GPUComputePipeline;
    private prefix_p!: GPUComputePipeline;
    // 3-phase scatter pipelines
    private scatter_local_even_p!: GPUComputePipeline;
    private scatter_local_odd_p!: GPUComputePipeline;
    private scatter_prefix_p!: GPUComputePipeline;
    private scatter_apply_even_p!: GPUComputePipeline;
    private scatter_apply_odd_p!: GPUComputePipeline;

    public subgroupSize!: number;

    private constructor() {}

    /**
     * Asynchronously creates and initializes a new GPURSSorter.
     */
    public static async create(device: GPUDevice, queue: GPUQueue): Promise<GPURSSorter> {
        console.debug("Searching for the maximum subgroup size...");
        const potentialSubgroupSizes = [16, 32, 16, 8, 1];

        for (const size of potentialSubgroupSizes) {
            console.debug(`Testing sorting with subgroup size ${size}`);
            try {
                const sorter = new GPURSSorter();
                await sorter.initializeWithSubgroupSize(device, size);
                const sortSuccess = await sorter.testSort(device, queue);
                if (sortSuccess) {
                    console.log(`Subgroup size ${size} works.`);
                    return sorter;
                }
            } catch (e) {
                console.warn(`Subgroup size ${size} failed during pipeline creation or test run.`, e);
            }
        }

        throw new Error("GPURSSorter::create() No working subgroup size was found. Unable to use sorter.");
    }

    /**
     * Initializes the sorter's pipelines and layouts for a given subgroup size.
     */
    private async initializeWithSubgroupSize(device: GPUDevice, sgSize: number) {
        this.subgroupSize = sgSize;

        this.bindGroupLayout = this.createBindGroupLayout(device);
        this.renderBindGroupLayout = GPURSSorter.createRenderBindGroupLayout(device);
        this.preprocessBindGroupLayout = GPURSSorter.createPreprocessBindGroupLayout(device);

        const pipelineLayout = device.createPipelineLayout({
            label: "radix sort pipeline layout",
            bindGroupLayouts: [this.bindGroupLayout],
        });

        const processedShaderCode = this.processShaderTemplate(radixSortShader);

        const shaderModule = device.createShaderModule({
            label: "Radix sort shader",
            code: processedShaderCode,
        });

        // Create all compute pipelines
        this.zero_p = await device.createComputePipelineAsync({
            label: "Zero the histograms",
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: "zero_histograms" },
        });
        this.histogram_p = await device.createComputePipelineAsync({
            label: "calculate_histogram",
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: "calculate_histogram" },
        });
        this.prefix_p = await device.createComputePipelineAsync({
            label: "prefix_histogram",
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: "prefix_histogram" },
        });
        // 3-phase scatter pipelines
        this.scatter_local_even_p = await device.createComputePipelineAsync({
            label: "scatter_local_even",
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: "scatter_local_even" },
        });
        this.scatter_local_odd_p = await device.createComputePipelineAsync({
            label: "scatter_local_odd",
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: "scatter_local_odd" },
        });
        this.scatter_prefix_p = await device.createComputePipelineAsync({
            label: "scatter_prefix_pass",
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: "scatter_prefix_pass" },
        });
        this.scatter_apply_even_p = await device.createComputePipelineAsync({
            label: "scatter_apply_even",
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: "scatter_apply_even" },
        });
        this.scatter_apply_odd_p = await device.createComputePipelineAsync({
            label: "scatter_apply_odd",
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: "scatter_apply_odd" },
        });
    }
 
    private processShaderTemplate(shaderCode: string): string {
        const histogram_sg_size = Math.max(1, this.subgroupSize | 0);

        const rs_sweep_0_size = Math.floor(RS_RADIX_SIZE / histogram_sg_size);
        const rs_sweep_1_size = Math.floor(rs_sweep_0_size / histogram_sg_size);
        const rs_sweep_2_size = Math.floor(rs_sweep_1_size / histogram_sg_size);

        const rs_smem_phase_2 = RS_RADIX_SIZE + RS_SCATTER_BLOCK_ROWS * SCATTER_WG_SIZE;
        const rs_mem_dwords = rs_smem_phase_2;
        const rs_mem_sweep_0_offset = 0;
        const rs_mem_sweep_1_offset = rs_mem_sweep_0_offset + rs_sweep_0_size;
        const rs_mem_sweep_2_offset = rs_mem_sweep_1_offset + rs_sweep_1_size;
        
        const constantDefinitions = `const histogram_sg_size: u32 = ${histogram_sg_size}u;
            const histogram_wg_size: u32 = ${HISTOGRAM_WG_SIZE}u;
            const rs_radix_log2: u32 = ${RS_RADIX_LOG2}u;
            const rs_radix_size: u32 = ${RS_RADIX_SIZE}u;
            const rs_keyval_size: u32 = ${RS_KEYVAL_SIZE}u;
            const rs_histogram_block_rows: u32 = ${RS_HISTOGRAM_BLOCK_ROWS}u;
            const rs_scatter_block_rows: u32 = ${RS_SCATTER_BLOCK_ROWS}u;
            const rs_mem_dwords: u32 = ${rs_mem_dwords}u;
            const rs_mem_sweep_0_offset: u32 = ${rs_mem_sweep_0_offset}u;
            const rs_mem_sweep_1_offset: u32 = ${rs_mem_sweep_1_offset}u;
            const rs_mem_sweep_2_offset: u32 = ${rs_mem_sweep_2_offset}u;
            `;

        let processedCode = shaderCode
            .replace(/{histogram_wg_size}/g, HISTOGRAM_WG_SIZE.toString())
            .replace(/{prefix_wg_size}/g, PREFIX_WG_SIZE.toString())
            .replace(/{scatter_wg_size}/g, SCATTER_WG_SIZE.toString());
        
        return constantDefinitions + processedCode;
    }

    /**
     * Runs a small test sort to verify the current configuration works.
     */
    private async testSort(device: GPUDevice, queue: GPUQueue): Promise<boolean> {
        const n = 8192;
        const scrambledData = new Float32Array(
            shuffleArray(Array.from({ length: n }, (_, i) => n - 1 - i))
        );
        const sortedData = new Float32Array(
            Array.from({ length: n }, (_, i) => i)
        ); 

        const sortStuff = this.createSortStuff(device, n);

        queue.writeBuffer(sortStuff.key_a, 0, scrambledData.buffer);

        const commandEncoder = device.createCommandEncoder({ label: "GPURSSorter test_sort" });
        this.recordSort(sortStuff, n, commandEncoder);
        queue.submit([commandEncoder.finish()]);

        await device.queue.onSubmittedWorkDone();
        
        const result = await this.downloadBuffer(device, queue, sortStuff.key_a, 'f32');

        for (let i = 0; i < n; i++) {
            if (result[i] !== sortedData[i]) {
                console.error(`Sort failed at index ${i}. Expected ${sortedData[i]}, got ${result[i]}`);
                return false;
            }
        }

        return true;
    }

    /**
     * Creates all the necessary buffers and bind groups for sorting a given number of points.
     */
    public createSortStuff(device: GPUDevice, numPoints: number): PointCloudSortStuff {
        const { key_a, key_b, payload_a, payload_b } = this.createKeyvalBuffers(device, numPoints, 4);
        const internal_mem = this.createInternalMemBuffer(device, numPoints);
        const source_indices = device.createBuffer({
            label: "Radix sort source indices",
            size: Math.max(1, numPoints) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        
        const { sorter_uni, sorter_dis, sorter_bg } = this.createBindGroup(
            device, numPoints, internal_mem, key_a, key_b, payload_a, payload_b
        );

        const sorter_render_bg = this.createRenderBindGroup(device, sorter_uni, payload_a);
        const sorter_bg_pre = this.createPreprocessBindGroup(device, sorter_uni, sorter_dis, key_a, payload_a, source_indices);

        return {
            numPoints,
            num_points: numPoints,
            sortedIndices: payload_a,
            indirectBuffer: sorter_dis,
            sorter_uni,
            sorter_dis,
            sorter_bg,
            sorter_render_bg,
            sorter_bg_pre,
            internal_mem,
            key_a,
            key_b,
            payload_a,
            payload_b,
            source_indices
        };
    }

    /**
     * Records sort commands using direct dispatch (known key count).
     * Each radix pass is: scatter_local -> scatter_prefix -> scatter_apply (3 separate compute passes)
     */
    public recordSort(sortStuff: SortedSplats, numPoints: number, encoder: GPUCommandEncoder): void {
        const radixStuff = sortStuff as PointCloudSortStuff;
        this.recordCalculateHistogram(radixStuff.sorter_bg, numPoints, encoder);
        this.recordPrefixHistogram(radixStuff.sorter_bg, 4, encoder);
        this.recordScatterKeys(radixStuff.sorter_bg, numPoints, encoder);
    }
    
    /**
     * Records sort commands using indirect dispatch (GPU-determined key count).
     * Same 3-phase scatter approach, using dispatchWorkgroupsIndirect.
     */
    public recordSortIndirect(sortStuff: SortedSplats, dispatchBuffer: GPUBuffer, encoder: GPUCommandEncoder): void {
        const radixStuff = sortStuff as PointCloudSortStuff;

        // Zero (indirect)
        {
            const pass = encoder.beginComputePass({ label: "RS::Zero (Indirect)" });
            pass.setBindGroup(0, radixStuff.sorter_bg);
            pass.setPipeline(this.zero_p);
            pass.dispatchWorkgroupsIndirect(dispatchBuffer, 0);
            pass.end();
        }

        // Histogram (indirect)
        {
            const pass = encoder.beginComputePass({ label: "RS::Histogram (Indirect)" });
            pass.setBindGroup(0, radixStuff.sorter_bg);
            pass.setPipeline(this.histogram_p);
            pass.dispatchWorkgroupsIndirect(dispatchBuffer, 0);
            pass.end();
        }

        // Prefix (direct, always 4 passes)
        this.recordPrefixHistogram(radixStuff.sorter_bg, 4, encoder);

        // 4 radix passes: even, odd, even, odd — each as 3-phase scatter
        this.recordScatterPassIndirect(radixStuff.sorter_bg, dispatchBuffer, true, encoder);  // pass 0 (even)
        this.recordScatterPassIndirect(radixStuff.sorter_bg, dispatchBuffer, false, encoder); // pass 1 (odd)
        this.recordScatterPassIndirect(radixStuff.sorter_bg, dispatchBuffer, true, encoder);  // pass 2 (even)
        this.recordScatterPassIndirect(radixStuff.sorter_bg, dispatchBuffer, false, encoder); // pass 3 (odd)
    }

    public recordSortIndirect_one(sortStuff: SortedSplats, dispatchBuffer: GPUBuffer, encoder: GPUCommandEncoder): void {
        // Alias to recordSortIndirect for backward compatibility
        this.recordSortIndirect(sortStuff, dispatchBuffer, encoder);
    }

    // Static methods for bind group layouts
    public static createRenderBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
        return device.createBindGroupLayout({
            label: "Radix Sort Render Bind Group Layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
            ],
        });
    }

    public static createPreprocessBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
        return device.createBindGroupLayout({
            label: "Radix Sort Preprocess Bind Group Layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // infos
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // keyval_a
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // payload_a
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // dispatch_buffer
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // source_indices
            ],
        });
    }

    public recordResetIndirectBuffer(indirectBuffer: GPUBuffer, uniformBuffer: GPUBuffer, queue: GPUQueue) {
        const zeroBuffer = new Uint32Array([0]);
        queue.writeBuffer(indirectBuffer, 0, zeroBuffer);
        queue.writeBuffer(uniformBuffer, 0, zeroBuffer);
    }

    // Private implementation methods
    private createBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
        return device.createBindGroupLayout({
            label: "Radix Sort Bind Group Layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ],
        });
    }

    private getScatterHistogramSizes(keysize: number): { scatter_blocks_ru: number, count_ru_histo: number } {
        const scatter_block_kvs = HISTOGRAM_WG_SIZE * RS_SCATTER_BLOCK_ROWS;
        const scatter_blocks_ru = Math.ceil(keysize / scatter_block_kvs);
        const count_ru_scatter = scatter_blocks_ru * scatter_block_kvs;
        
        const histo_block_kvs = HISTOGRAM_WG_SIZE * RS_HISTOGRAM_BLOCK_ROWS;
        const histo_blocks_ru = Math.ceil(count_ru_scatter / histo_block_kvs);
        const count_ru_histo = histo_blocks_ru * histo_block_kvs;

        return { scatter_blocks_ru, count_ru_histo };
    }

    private createKeyvalBuffers(device: GPUDevice, keysize: number, bytesPerPayloadElem: number): { key_a: GPUBuffer, key_b: GPUBuffer, payload_a: GPUBuffer, payload_b: GPUBuffer } {
        const keys_per_workgroup = HISTOGRAM_WG_SIZE * RS_HISTOGRAM_BLOCK_ROWS;
        const count_ru_histo =
            (Math.floor((keysize + keys_per_workgroup) / keys_per_workgroup) + 1) * keys_per_workgroup;

        const paddedKeySize = count_ru_histo * Float32Array.BYTES_PER_ELEMENT;

        const buffer_a = device.createBuffer({
            label: "Radix data buffer a",
            size: paddedKeySize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        const buffer_b = device.createBuffer({
            label: "Radix data buffer b",
            size: paddedKeySize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        if (bytesPerPayloadElem !== 4) {
            console.warn("Currently only 4-byte payloads are fully supported.");
        }
        const payloadSize = Math.max(1, keysize * bytesPerPayloadElem);
        const payload_a = device.createBuffer({
            label: "Radix payload buffer a",
            size: payloadSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        const payload_b = device.createBuffer({
            label: "Radix payload buffer b",
            size: payloadSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        return { key_a: buffer_a, key_b: buffer_b, payload_a, payload_b };
    }

    private createInternalMemBuffer(device: GPUDevice, keysize: number): GPUBuffer {
        const { scatter_blocks_ru } = this.getScatterHistogramSizes(keysize);
        const histo_size = RS_RADIX_SIZE * Uint32Array.BYTES_PER_ELEMENT;
        // Layout: histograms + partitions + prefix areas
        // histograms: RS_KEYVAL_SIZE rows
        // partitions: (scatter_blocks_ru + 1) rows  (+1 for safety addition from preprocess)
        // prefix:     (scatter_blocks_ru + 1) rows
        const internal_size = (RS_KEYVAL_SIZE + (scatter_blocks_ru + 1) * 2) * histo_size;
  
        return device.createBuffer({
            label: "Internal radix sort buffer",
            size: internal_size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
    }

    private createBindGroup(
        device: GPUDevice,
        keysize: number,
        internal_mem_buffer: GPUBuffer,
        keyval_a: GPUBuffer,
        keyval_b: GPUBuffer,
        payload_a: GPUBuffer,
        payload_b: GPUBuffer
    ): { sorter_uni: GPUBuffer, sorter_dis: GPUBuffer, sorter_bg: GPUBindGroup } {
        const { scatter_blocks_ru, count_ru_histo } = this.getScatterHistogramSizes(keysize);

        const uniform_infos: GeneralInfo = {
            keys_size: keysize,
            padded_size: count_ru_histo,
            passes: 4,
            even_pass: 0,
            odd_pass: 0,
        };
        const uniform_buffer = device.createBuffer({
            label: "Radix uniform buffer",
            size: 5 * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST| GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });
        new Uint32Array(uniform_buffer.getMappedRange()).set([
            uniform_infos.keys_size,
            uniform_infos.padded_size,
            uniform_infos.passes,
            uniform_infos.even_pass,
            uniform_infos.odd_pass,
        ]);
        uniform_buffer.unmap();

        const dispatch_infos: IndirectDispatch = {
            dispatch_x: scatter_blocks_ru,
            dispatch_y: 1,
            dispatch_z: 1,
        };
        const dispatch_buffer = device.createBuffer({
            label: "Dispatch indirect buffer",
            size: 3 * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
            mappedAtCreation: true,
        });
        new Uint32Array(dispatch_buffer.getMappedRange()).set([
            dispatch_infos.dispatch_x,
            dispatch_infos.dispatch_y,
            dispatch_infos.dispatch_z,
        ]);
        dispatch_buffer.unmap();

        const bind_group = device.createBindGroup({
            label: "Radix bind group",
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: uniform_buffer } },
                { binding: 1, resource: { buffer: internal_mem_buffer } },
                { binding: 2, resource: { buffer: keyval_a } },
                { binding: 3, resource: { buffer: keyval_b } },
                { binding: 4, resource: { buffer: payload_a } },
                { binding: 5, resource: { buffer: payload_b } },
            ],
        });

        return { sorter_uni: uniform_buffer, sorter_dis: dispatch_buffer, sorter_bg: bind_group };
    }

    private createRenderBindGroup(device: GPUDevice, general_infos: GPUBuffer, payload_a: GPUBuffer): GPUBindGroup {
        return device.createBindGroup({
            label: "Render bind group",
            layout: this.renderBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: general_infos } },
                { binding: 4, resource: { buffer: payload_a } },
            ],
        });
    }
    
    private createPreprocessBindGroup(
        device: GPUDevice,
        uniform_buffer: GPUBuffer,
        dispatch_buffer: GPUBuffer,
        keyval_a: GPUBuffer,
        payload_a: GPUBuffer,
        source_indices: GPUBuffer
    ): GPUBindGroup {
        return device.createBindGroup({
            label: "Preprocess bind group",
            layout: this.preprocessBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: uniform_buffer } },
                { binding: 1, resource: { buffer: keyval_a } },
                { binding: 2, resource: { buffer: payload_a } },
                { binding: 3, resource: { buffer: dispatch_buffer } },
                { binding: 4, resource: { buffer: source_indices } },
            ],
        });
    }

    private recordCalculateHistogram(bind_group: GPUBindGroup, keysize: number, encoder: GPUCommandEncoder) {
        const { count_ru_histo } = this.getScatterHistogramSizes(keysize);
        const histo_block_kvs = HISTOGRAM_WG_SIZE * RS_HISTOGRAM_BLOCK_ROWS;
        const hist_blocks_ru = Math.ceil(count_ru_histo / histo_block_kvs);

        // Pass A: Zero
        {
            const pass = encoder.beginComputePass({ label: "RS::Zero" });
            pass.setBindGroup(0, bind_group);
            pass.setPipeline(this.zero_p);
            pass.dispatchWorkgroups(hist_blocks_ru, 1, 1);
            pass.end();
        }

        // Pass B: Histogram
        {
            const pass = encoder.beginComputePass({ label: "RS::Histogram" });
            pass.setBindGroup(0, bind_group);
            pass.setPipeline(this.histogram_p);
            pass.dispatchWorkgroups(hist_blocks_ru, 1, 1);
            pass.end();
        }
    }

    private recordPrefixHistogram(bind_group: GPUBindGroup, passes: number, encoder: GPUCommandEncoder) {
        const pass = encoder.beginComputePass({ label: "Radix Sort :: Prefix Sum Pass" });
        pass.setPipeline(this.prefix_p);
        pass.setBindGroup(0, bind_group);
        pass.dispatchWorkgroups(passes, 1, 1);
        pass.end();
    }

    /**
     * Records the 4 radix scatter passes using direct dispatch.
     * Each radix pass is 3 compute passes: scatter_local -> scatter_prefix -> scatter_apply
     */
    private recordScatterKeys(bind_group: GPUBindGroup, keysize: number, encoder: GPUCommandEncoder) {
        const { scatter_blocks_ru } = this.getScatterHistogramSizes(keysize);

        const recordScatterPass = (
            localPipeline: GPUComputePipeline,
            applyPipeline: GPUComputePipeline,
            labelPrefix: string
        ) => {
            // Phase 1: scatter_local
            {
                const pass = encoder.beginComputePass({ label: `${labelPrefix}::Local` });
                pass.setBindGroup(0, bind_group);
                pass.setPipeline(localPipeline);
                pass.dispatchWorkgroups(scatter_blocks_ru, 1, 1);
                pass.end();
            }
            // Phase 2: scatter_prefix_pass (single workgroup scans all partitions)
            {
                const pass = encoder.beginComputePass({ label: `${labelPrefix}::Prefix` });
                pass.setBindGroup(0, bind_group);
                pass.setPipeline(this.scatter_prefix_p);
                pass.dispatchWorkgroups(1, 1, 1);
                pass.end();
            }
            // Phase 3: scatter_apply
            {
                const pass = encoder.beginComputePass({ label: `${labelPrefix}::Apply` });
                pass.setBindGroup(0, bind_group);
                pass.setPipeline(applyPipeline);
                pass.dispatchWorkgroups(scatter_blocks_ru, 1, 1);
                pass.end();
            }
        };

        // 4 radix passes: even, odd, even, odd
        recordScatterPass(this.scatter_local_even_p, this.scatter_apply_even_p, "RS::Scatter0_even");
        recordScatterPass(this.scatter_local_odd_p,  this.scatter_apply_odd_p,  "RS::Scatter1_odd");
        recordScatterPass(this.scatter_local_even_p, this.scatter_apply_even_p, "RS::Scatter2_even");
        recordScatterPass(this.scatter_local_odd_p,  this.scatter_apply_odd_p,  "RS::Scatter3_odd");
    }

    /**
     * Records a single 3-phase scatter pass using indirect dispatch.
     */
    private recordScatterPassIndirect(
        bind_group: GPUBindGroup,
        dispatchBuffer: GPUBuffer,
        isEven: boolean,
        encoder: GPUCommandEncoder
    ) {
        const localPipeline = isEven ? this.scatter_local_even_p : this.scatter_local_odd_p;
        const applyPipeline = isEven ? this.scatter_apply_even_p : this.scatter_apply_odd_p;
        const label = isEven ? "even" : "odd";

        // Phase 1: scatter_local (indirect)
        {
            const pass = encoder.beginComputePass({ label: `RS::ScatterLocal_${label} (Indirect)` });
            pass.setBindGroup(0, bind_group);
            pass.setPipeline(localPipeline);
            pass.dispatchWorkgroupsIndirect(dispatchBuffer, 0);
            pass.end();
        }
        // Phase 2: scatter_prefix_pass (always 1 workgroup, direct dispatch)
        {
            const pass = encoder.beginComputePass({ label: `RS::ScatterPrefix_${label} (Indirect)` });
            pass.setBindGroup(0, bind_group);
            pass.setPipeline(this.scatter_prefix_p);
            pass.dispatchWorkgroups(1, 1, 1);
            pass.end();
        }
        // Phase 3: scatter_apply (indirect)
        {
            const pass = encoder.beginComputePass({ label: `RS::ScatterApply_${label} (Indirect)` });
            pass.setBindGroup(0, bind_group);
            pass.setPipeline(applyPipeline);
            pass.dispatchWorkgroupsIndirect(dispatchBuffer, 0);
            pass.end();
        }
    }

    /**
     * Helper function to download buffer data from the GPU.
     */
    private async downloadBuffer(device: GPUDevice, queue: GPUQueue, buffer: GPUBuffer, type: 'f32' | 'u32'): Promise<Float32Array | Uint32Array> {
        const downloadBuffer = device.createBuffer({
            label: "Download buffer",
            size: buffer.size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        const commandEncoder = device.createCommandEncoder({ label: "Copy encoder" });
        commandEncoder.copyBufferToBuffer(buffer, 0, downloadBuffer, 0, buffer.size);
        queue.submit([commandEncoder.finish()]);

        await downloadBuffer.mapAsync(GPUMapMode.READ);
        const data = downloadBuffer.getMappedRange();
        
        let result: Float32Array | Uint32Array;
        if (type === 'f32') {
            result = new Float32Array(data.slice(0));
        } else {
            result = new Uint32Array(data.slice(0));
        }

        downloadBuffer.unmap();
        downloadBuffer.destroy();

        return result;
    }
}
