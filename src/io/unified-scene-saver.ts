// 场景与模型保存入口模块
// 提供 saveUnifiedScene 主接口，用于保存场景数据及模型文件至指定文件夹
// 使用示例类似于: await saveUnifiedScene({ scenes, folderHandle, meta });

export interface SaveUnifiedSceneModel {
    /** 物体唯一ID，用于关联关键帧 */
    id: string;
    /** 用于 UI 展示的名称，可选 */
    name?: string;
    /** 类型标签，用于区分不同对象类型（例如 fileModel/recordingCamera/primitive.cube 等） */
    typeTag: string;
    /** 变换TRS：[translate, rotate, scale] */
    trs: [number[], number[], number[]];
    /** 可选，模型类型、后缀等元信息 */
    type?: string;
    /** 高斯模型参数 */
    gaussianParams?: any;
    /** 录制相机参数 */
    cameraParams?: any;
    /** 类型化参数，用于保存特殊对象（如相机、原生几何体等）所需的额外信息 */
    params?: Record<string, any>;
    /** 源模型文件句柄或Blob或File，必须能读取实际数据内容。对于非文件对象可省略 */
    originFile?: File | Blob | FileSystemFileHandle;
    /** 如果存在外部资源文件，保存时的目标文件名 */
    assetName?: string;
}

export interface SaveUnifiedSceneKeyframe {
    objectId: string;
    frame: number;
    trs: [number[], number[], number[]];
    gaussianParams?: any;
    cameraParams?: any;
}

export interface SaveUnifiedSceneSceneEntry {
    models: SaveUnifiedSceneModel[];
    keyframes?: SaveUnifiedSceneKeyframe[];
    view?: 'left' | 'right' | string;
    [k: string]: any;
}

export interface SaveUnifiedSceneParams {
    /** 天空盒设置 */
    skyboxEnabled: boolean;
    /** 天空盒URL */
    // skyboxUrl: string;
    /** 多场景结构 */
    scenes: SaveUnifiedSceneSceneEntry[];
    /** 用户授权的目标文件夹（场景文件夹） */
    folderHandle: FileSystemDirectoryHandle;
    /** 额外信息如createdAt/author等，用于 scene.json meta 字段 */
    meta?: any;
    /** 相机参数 */
    cameraParams?: any;
    /** 总帧数 */
    totalFrames?: number;
}

/**
 * 验证文件夹句柄是否仍然有效
 */
async function verifyFolderHandle(folderHandle: FileSystemDirectoryHandle): Promise<void> {
    try {
        // 尝试查询权限以验证句柄是否有效
        const permission = await folderHandle.queryPermission({ mode: 'readwrite' });
        if (permission !== 'granted') {
            // 如果权限不是 granted，尝试请求权限
            const requested = await folderHandle.requestPermission({ mode: 'readwrite' });
            if (requested !== 'granted') {
                throw new Error('文件夹权限被拒绝或已失效');
            }
        }
        // 尝试访问文件夹名称以验证句柄是否仍然有效
        // 如果文件夹被删除，访问 name 属性可能会失败
        const _ = folderHandle.name;
    } catch (error) {
        const errorMessage = (error as Error).message || String(error);
        if (errorMessage.includes('could not be found') ||
            errorMessage.includes('not found') ||
            errorMessage.includes('权限') ||
            errorMessage.includes('permission')) {
            throw new Error(`文件夹句柄已失效：${errorMessage}`);
        }
        throw error;
    }
}

/**
 * 清空文件夹内容（删除所有文件和子文件夹）
 */
async function clearFolder(folderHandle: FileSystemDirectoryHandle): Promise<void> {
    try {
        // 遍历文件夹中的所有条目
        // FileSystemDirectoryHandle 是异步可迭代的，返回 [name, handle] 键值对
        const entries: Array<[string, FileSystemFileHandle | FileSystemDirectoryHandle]> = [];
        for await (const [name, handle] of folderHandle as any) {
            entries.push([name, handle]);
        }

        // 删除所有条目
        for (const [entryName, handle] of entries) {
            try {
                if (handle.kind === 'file') {
                    // 删除文件
                    await folderHandle.removeEntry(entryName);
                } else if (handle.kind === 'directory') {
                    // 递归删除子文件夹
                    await clearFolder(handle as FileSystemDirectoryHandle);
                    await folderHandle.removeEntry(entryName);
                }
            } catch (error) {
                const errorMessage = (error as Error).message || String(error);
                console.warn(`[Scene] 删除条目 ${entryName} 时出现错误:`, errorMessage);
            }
        }
    } catch (error) {
        const errorMessage = (error as Error).message || String(error);
        // 如果文件夹为空或不存在，忽略错误
        if (!errorMessage.includes('could not be found') &&
            !errorMessage.includes('not found')) {
            console.warn('[Scene] 清空文件夹时出现错误:', errorMessage);
            throw new Error(`清空文件夹失败: ${errorMessage}`);
        }
    }
}

/**
 * 保存场景与模型文件到指定目录
 * @param params scenes数组、目标文件夹句柄、可选meta信息
 * @throws 拷贝或写入失败会抛出异常，请在上层捕获并提示用户
 */
export async function saveUnifiedScene(params: SaveUnifiedSceneParams): Promise<void> {
    const { skyboxEnabled, scenes, folderHandle, meta } = params;

    // 在开始保存前验证文件夹句柄有效性
    await verifyFolderHandle(folderHandle);

    // 如果文件夹内有内容，清空文件夹
    await clearFolder(folderHandle);

    // 去重统计所有需要拷贝的模型信息
    const copiedFileNames = new Set<string>();
    for (const scene of scenes) {
        if (!scene || !Array.isArray(scene.models)) continue;
        for (const model of scene.models) {
            if (!model) continue;
            if (!model.id) throw new Error('模型必须包含id字段');
            if (!model.originFile) continue; // 非文件型对象无需写入
            // const fileName = model.assetName || model.name;
            const fileName = model.name || model.assetName;
            if (!fileName) throw new Error(`对象 ${model.id} 缺少保存外部资源所需的名称`);
            if (copiedFileNames.has(fileName)) continue; // 同名文件只写一次
            copiedFileNames.add(fileName);
            // 拷贝文件：必须能从 originFile 读取内容
            let fileData: Blob;
            if (model.originFile instanceof File || model.originFile instanceof Blob) {
                fileData = model.originFile;
            } else if ('getFile' in model.originFile && typeof model.originFile.getFile === 'function') {
                fileData = await model.originFile.getFile();
            } else {
                throw new Error(`无法识别模型文件originFile类型: ${model.name}`);
            }

            try {
                // 在写入每个文件前再次验证文件夹句柄有效性（防止在长时间操作中失效）
                await verifyFolderHandle(folderHandle);
                // 写入目标目录下
                const destHandle = await folderHandle.getFileHandle(fileName, { create: true });
                const writable = await destHandle.createWritable();
                await writable.write(fileData);
                await writable.close();
            } catch (error) {
                const errorMessage = (error as Error).message || String(error);
                console.error('[Scene] 写入模型文件失败:', fileName, error);
                // 如果是文件夹失效错误，直接抛出
                if (errorMessage.includes('could not be found') ||
                    errorMessage.includes('not found') ||
                    errorMessage.includes('已失效') ||
                    errorMessage.includes('权限')) {
                    throw error;
                }
                // 其他错误也抛出，确保上层能够处理
                throw new Error(`写入模型文件 ${fileName} 失败: ${errorMessage}`);
            }
        }
    }
    // 构造 scene.json 内容
    const sceneJson = {
        version: 1,
        meta: meta || { createdAt: new Date().toISOString(), app: 'VisionaryEditor' },
        skyboxEnabled: skyboxEnabled || false,
        camera: params.cameraParams || { position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1], fov: 60, nearPlane: 0.1, farPlane: 1000 },
        totalFrames: params.totalFrames || 100,
        scenes: scenes.map(scene => ({
            ...scene,
            // 为兼容性防止存储不必要runtime字段
            models: scene.models.map(({ originFile, ...rest }) => rest)
        }))
    };
    // 在写入 scene.json 前再次验证文件夹句柄有效性
    await verifyFolderHandle(folderHandle);

    // 写 scene.json
    try {
        console.log('[Scene] 写入 scene.json:', sceneJson);
        const sceneHandle = await folderHandle.getFileHandle('scene.json', { create: true });
        const sceneWritable = await sceneHandle.createWritable();
        await sceneWritable.write(JSON.stringify(sceneJson, null, 2));
        await sceneWritable.close();
    } catch (error) {
        const errorMessage = (error as Error).message || String(error);
        console.error('[Scene] 写入 scene.json 失败:', error);
        throw new Error(`写入场景配置文件失败: ${errorMessage}`);
    }
}
