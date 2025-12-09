# Showcase 使用说明（示例主页）

面向使用 VisionaryCore 的开发者，示例主页展示了如何在浏览器里加载多种模型（含 ONNX 4DGS），并带有基础相机动画。假设你已克隆本仓库、安装好 Node 依赖，并将 `src/showcase` 复制到项目根目录或 `public/showcase` 下。

## 目录结构
- `index.html`：示例入口页面。
- `scripts/ShowcaseScene.ts`：核心逻辑，封装三场景切换与渲染。
- `scripts/sceneConfigs.ts`：场景模型与相机配置，主要改这里即可替换资源/相机。
- `scripts/main.ts`：启动入口，创建 `ShowcaseScene` 并绑定 UI。
- `styles/showcase.css`：样式。

## 前置与运行
1) 安装依赖（仓库根目录）：
```bash
npm install
```
2) WebGPU 环境：需支持 WebGPU 的浏览器（Chrome 121+ 或 Canary，必要时启用 `chrome://flags` 的 WebGPU）。
3) 资源准备（见下一节）。
4) 运行开发服务器（根目录）：
```bash
npm run dev
```
默认端口 3000，对应 URL：
- 当前目录（`demo/showcase`）已在仓库内，启动后直接访问 `http://localhost:3000/demo/showcase/`
- 如果你将 showcase 拷贝到 `public/showcase/`，访问 `http://localhost:3000/showcase/`

> 若提示 WebGPU/ORT 相关跨域或 COOP/COEP 问题，请确保本地服务器返回合适的跨源隔离头；Vite 默认已开启。

## 模型与文件放置
当前配置引用的资源（来自 `scripts/sceneConfigs.ts`），可选择下载并按原路径放置，或修改路径指向你自己的模型：
- `/models/册方彝.glb`
- `/models/fox.ply`
- `/models/dyn/mutant.onnx`
- `/models/谷纹青玉璧.glb`
- `/models/dyn/hellwarrior.onnx`
- `/models/白玉卧鹿.glb`
- `/models/dyn/trex.onnx`
- `/models/qiewu/gaussianA.onnx` ... `gaussianG.onnx`

放置建议：
- 统一放到 `public/models/`（或 `showcase/models/`，保持 URL 对应）。
- 如果改路径，修改 `sceneConfigs.ts` 中相应的 `url`。

ONNX Runtime 依赖的 wasm 路径：
- `ShowcaseScene` 调用 `initOrtEnvironment(getDefaultOrtWasmPaths())`，默认路径 `/src/ort/`。运行 `npm install` 后脚本会把 ORT 运行时拷贝到 `src/ort`，开发和预览都能直接访问。若你单独部署 showcase，请同步拷贝 `src/ort` 到可访问位置，或在代码里改 `wasmPaths`。

## 常见操作
- 更换模型：在 `sceneConfigs.ts` 替换某个 `url` + `loadOptions`（`type: 'onnx' | 'glb' | 'ply' | ...`）。
- 调整相机：修改 `getScene2CameraViews()` 或 `SCENE3_CAMERA_BASE / SCENE3_CAMERA_SWING`。
- 修改轮播：调整 `SCENE1_CAROUSEL_ITEMS` 的条目、间距、旋转速度（对应 `ShowcaseScene` 内的 `carouselSpacing` 等）。

## 扩展性建议
如果你想添加自定义场景（例如 `loadScene4`）：
1) 在 `ShowcaseScene` 内新增方法 `async loadScene4()`，调用 `loadUnifiedModel` 加载你的资源并布置到 `scene`。
2) 在 `switchToScene` 增加分支调用 `loadScene4`。
3) 在 UI（`main.ts` 或页面按钮）增加触发逻辑。

为了配置更友好，也可以将模型与相机配置完全数据化：
- 在 `sceneConfigs.ts` 新增一个通用列表，比如 `SCENE4_CONFIGS`（包含模型 URL、变换、相机初始值），然后在 `ShowcaseScene` 里按配置循环生成模型，减少硬编码。

## 若要自带模型包分发
- 将 `public/models`（或你自定义的目录）与 `src/ort` 一起打包/上线，保持与 `sceneConfigs.ts` 中的 URL 一致。
- 如果希望用户下载后本地打开，可提供压缩包结构示例：
  ```
  showcase/
    index.html
    scripts/
    styles/
    models/            # 各类 glb/ply/onnx
    ort/               # ort-wasm-simd-threaded.* 等
  ```

这样即可直接在本地或 CDN 上作为示例主页使用，并能快速替换/新增场景以验证 VisionaryCore 的加载与渲染。

## 像 Spark 一样用 importmap 引用（CDN 方案）
我们新增了库构建脚本，产出单文件 ES 模块，便于直接用 importmap：
1) 构建库：
```bash
npm run build:core-lib
```
输出位置：`dist-core/visionary-core.es.js`，并自动拷贝 ORT 运行时到 `dist-core/ort/`。
2) 将以下文件上传到你的 CDN（需同域或具备正确 CORS/COOP/COEP）：
   - `dist-core/visionary-core.es.js`
   - `dist-core/ort/*`（四个 ort-wasm-simd-threaded 文件）
   - 你用于拉起 ORT 的 dummy 模型（`/models/onnx_dummy.onnx`，需自行提供并放在可访问路径）
   - 业务模型资源（glb/ply/onnx），路径与页面 import URL 对齐
3) 页面示例（importmap）：
```html
<script type="importmap">
{
  "imports": {
    "three": "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.171.0/three.module.js",
    "onnxruntime-web/webgpu": "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort-webgpu.min.js",
    "@visionary/core": "https://your.cdn.com/visionary-core.es.js"
  }
}
</script>
```
4) 使用时：
```js
import { initThreeContext, loadUnifiedModel } from '@visionary/core';
// 确保 initThreeContext 能访问到 /models/onnx_dummy.onnx 与 /ort/*
```
注意：
- 需浏览器支持 WebGPU，并且站点返回 COOP/COEP，以便 ORT + WebGPU 正常工作。
- `visionary-core.es.js` 将 `three` 与 `onnxruntime-web/webgpu` 视为外部依赖，必须由 importmap 提供。

