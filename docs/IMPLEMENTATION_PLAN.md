# Quant Benchmark Lab — Implementation & Execution Plan

Trạng thái: kế hoạch triển khai, chưa có code benchmark hoặc kết quả đo. Workspace hiện chỉ có `AGENTS.md`; tài liệu này không giả định đã có toolchain, model hay GPU khả dụng.

Mục tiêu: benchmark small open-weight LLM trên RTX 3050 4GB, giải thích được tradeoff quality, tốc độ, latency và VRAM bằng dữ liệu có nguồn gốc và protocol tái lập được.

## 1. Quyết định kiến trúc và phạm vi

### MVP bắt buộc

- Python 3.11+, `pyproject.toml`, `uv.lock`, CLI tên `qbl`.
- Hai backend HTTP: Ollama `/api/generate` và llama.cpp `llama-server` `/completion`; dùng native API để lấy timing.
- Một máy, một GPU, một request đang chạy, một model resident, một slot/sequence. Benchmark HTTP qua loopback.
- Linux native là môi trường tham chiếu. Môi trường khác có thể chạy mock/CPU; không trộn kết quả vào cùng campaign GPU.
- Ba experiment độc lập; cùng artifact GGUF cho so sánh engine; không dùng model tag tải sẵn làm bằng chứng artifact tương đương.
- JSONL/JSON cho raw results, CSV + Markdown + PNG cho reports. Không cần database, dashboard, plugin registry hay distributed workers.
- Quality MVP: exact match, numeric answer và structured extraction. Coding tests và LLM judge là phần mở rộng.
- Mock chỉ kiểm tra harness; mọi fixture/report giả lập phải mang `synthetic=true` và không được xuất thành kết quả benchmark thật.

### Cấu trúc repository đích

```text
pyproject.toml
uv.lock
README.md
configs/
  models.yaml
  runtime.yaml
  protocol.yaml
  experiments/{quantization,engine,model}.yaml
locks/{toolchain,models,workloads}.json
src/quant_benchmark_lab/
  cli.py
  config.py
  schema.py
  environment.py
  artifacts.py
  prompts.py
  protocol.py
  runner.py
  storage.py
  metrics.py
  backends/{base,fake,ollama,llamacpp,streaming}.py
  monitoring/gpu.py
  quality/{dataset,scorers,evaluator}.py
  reporting/{aggregate,reporter,plots}.py
scripts/{bootstrap_llama.sh,prepare_models.py}
data/{performance,quality}/
tests/{unit,contract,integration,fixtures}/
results/raw/<campaign_id>/
reports/<campaign_id>/
docs/{IMPLEMENTATION_PLAN,methodology,reproduction,limitations}.md
```

`BenchmarkRunner` chỉ điều phối lifecycle và request; backend sở hữu process/API/protocol; `GPUMonitor` lấy mẫu riêng; `QualityEvaluator` chấm offline; `Reporter` chỉ đọc dữ liệu đã lưu. Không để backend tự chạy repetitions, chấm điểm hoặc ghi report.

Runtime dependencies: `httpx`, `pydantic`, `PyYAML`, `psutil`, `nvidia-ml-py`, `numpy`, `matplotlib`. CLI dùng `argparse`, report Markdown dùng template đơn giản. Development: `pytest`, `ruff`. Dependencies chuyển đổi model nằm trong nhóm `prepare`, pin riêng theo toolchain, không bắt runtime phải cài PyTorch.

## 2. Methodology phải khóa trước khi benchmark chính thức

### 2.1 Ba experiment

| Experiment | Biến thay đổi | Điều kiện giữ cố định | Matrix MVP |
|---|---|---|---|
| E1 Quantization | F16, Q8_0, Q4_K_M | Cùng source revision, model, tokenizer, llama.cpp build, template, runtime, workload | Qwen2.5-0.5B-Instruct; thêm 1.5B nếu pilot cho phép |
| E2 Engine | Ollama, llama.cpp | Cùng SHA-256 GGUF Q4_K_M, prompt bytes, tokenizer behavior, sampling, context, offload, workload | Qwen2.5-1.5B-Instruct |
| E3 Model | Model/checkpoint/size | llama.cpp build, Q4_K_M, runtime, task content, protocol | Qwen2.5-0.5B, 1.5B, 3B Instruct; Qwen3-4B non-thinking làm ứng viên gần giới hạn |

Các checkpoint trên là lựa chọn khởi đầu có model card chính thức: [0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [3B](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B). Phải pin revision SHA, license, tokenizer và template trước khi tải chính thức.

- E3 đo sản phẩm model + tokenizer + template phù hợp; không diễn giải thành tác động riêng của số parameter. Cùng tên quant cũng không đồng nghĩa cùng bits/weight thực tế.
- Ghi GGUF bytes, tensor-type distribution và bytes/parameter. Mỗi quant được tạo trực tiếp từ cùng F16 GGUF, không requantize từ Q8 sang Q4.
- F16 là baseline FP16, không gọi là một thuật toán quantization.
- Main comparison yêu cầu full GPU offload, gồm các tensor/layer mà kiến trúc và backend hỗ trợ offload. CPU tokenization/sampling không bị coi là CPU layer offload. Ghi cả yêu cầu và trạng thái thực tế.
- OOM hoặc tự động offload một phần sang CPU phải thành `infeasible` hoặc `invalid_configuration`; không âm thầm giảm context, quant hay số layer. Hybrid offload là experiment sau.
- Định nghĩa “gần giới hạn”: model full-offload có peak device VRAM ≥80% tổng VRAM, đo trong môi trường idle đã ổn định và có quy trách nhiệm process; không lấy VRAM của ứng dụng khác để đạt ngưỡng.
- Pilot thử context chung 4096; nếu chưa có model đạt ngưỡng, thử 8192 cho toàn bộ matrix trước khi freeze. Ghi rõ đây là configured context, khác input length thực tế. Nếu vẫn thiếu model vừa fit vừa đạt ngưỡng, campaign chưa đạt yêu cầu E3; phải bổ sung ứng viên và làm lại pilot, không tuyên bố hoàn thành bằng ước lượng kích thước file.

### 2.2 Generation và runtime profile

Profile khởi đầu: `temperature=0`, `seed=42`, `top_k=1`, `top_p=1`, `min_p=0`, `repeat_penalty=1`, `repeat_last_n=0`, presence/frequency penalties bằng 0, Mirostat/dynamic temperature/DRY/XTC/speculative decoding tắt nếu backend có các cơ chế này. Đối với engine không expose một tham số: chứng minh nó bị vô hiệu hoặc không áp dụng ở bản pin; không gửi option rồi mặc định nó được tôn trọng.

- `context_size=4096` ban đầu; performance `max_tokens=128`, quality `max_tokens=256`.
- Thinking: `not_applicable` với model không reasoning; `disabled` với Qwen3 qua template non-thinking đã pin. Trong raw completion, không chỉ dựa vào một cờ API để giả định thinking tắt.
- Stop strings được lưu đầy đủ theo model; E1/E2 phải giống nhau. EOS tự nhiên được giữ; lưu stop reason và actual output tokens. Không dùng padding hoặc bỏ EOS để giả vờ output luôn dài 128 token.
- Threads bằng số physical CPU cores phát hiện lúc preflight, ghi thành số cụ thể trong resolved config. Batch size 128; llama.cpp microbatch 128 nếu hỗ trợ. Ghi cấu hình nội bộ không thể điều khiển của Ollama như limitation.
- KV-cache K/V dùng F16; flash attention tắt ở baseline nếu cả hai engine hỗ trợ; không đổi riêng một engine để chữa OOM. mmap bật, mlock tắt nếu khả dụng; mọi khác biệt phải được ghi lại.
- Tắt automatic context/offload fitting nếu backend expose; nếu không, kiểm tra effective configuration và reject fallback. Ollama dùng `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=1`; llama-server dùng một slot. Không có conversation history truyền lại giữa requests.
- Greedy + seed không đảm bảo bitwise-identical output giữa engine, GPU kernels hoặc phiên bản. Không dùng “cùng seed” làm bằng chứng so sánh tương đương.
- E2 kết luận về hai serving stacks với version/build đã pin; không khẳng định chỉ đo overhead HTTP vì implementation/kernel bên trong có thể khác.

### 2.3 Prompt và token accounting

Harness render prompt đúng một lần, lưu UTF-8 bytes/hash. Ollama dùng `raw=true`; llama.cpp dùng native completion, tránh engine áp lại chat template. Ollama có hỗ trợ raw prompt và import GGUF local qua `FROM`; vẫn phải xác nhận blob đã import đúng artifact. [Ollama generate](https://docs.ollama.com/api/generate), [Importing a model](https://docs.ollama.com/import).

- E1/E2: cùng prompt bytes, BOS/EOS policy, special-token parsing, stop strings, input token count; test golden tokenization từ GGUF. Nếu Ollama không expose token IDs, lưu log/count và kiểm tra hành vi BOS với prompt đặc biệt; ghi giới hạn quan sát, không khai đã chứng minh toàn bộ IDs.
- E3: cùng task text, dùng template chính thức tương ứng. Không pad các model để ép bằng token count; report token counts và latency trên cùng task.
- Tách `input_tokens_total`, `input_tokens_cached`, `input_tokens_evaluated`, `output_tokens_engine`. Không đếm chunk HTTP thành token. Không dùng tokenizer khác để thay token count engine mà không gắn nhãn estimate.
- Nếu engine không cung cấp token count/cache semantics đủ rõ ở bản pin, request không đủ điều kiện cho prefill comparison. Không tự suy ra bằng tốc độ.
- Reject prompt khi input + output budget vượt context; không chấp nhận silent truncation hoặc context shifting.

### 2.4 Metric definitions

Mọi wall-clock duration dùng `perf_counter_ns()`; UTC chỉ làm timestamp audit. Lưu raw native timing và normalized unit riêng.

| Metric chuẩn | Định nghĩa |
|---|---|
| `model_load_wall_ms` | Từ yêu cầu activation khi model chưa resident đến readiness xác nhận. Ollama: explicit preload request hoàn tất; llama.cpp: spawn server đến health ready. Đây là operational load-to-ready, có initialization khác nhau giữa stack. |
| `model_load_native_ms` | Engine báo riêng nếu có semantic đã xác minh. Có thể null kèm lý do; không thay thế bằng wall time hoặc so sánh như cùng ranh giới đo. |
| `ttft_stream_ms` | Từ trước HTTP send đến chunk đầu tiên có text sinh ra, kể cả whitespace; bỏ heartbeat/metadata/empty text. Đây là client-observed first-content latency, proxy cho TTFT, có buffering. |
| `prefill_tok_s` | Số input token thực sự được evaluate / native prefill seconds. Không lấy toàn prompt chia thời gian chỉ xử lý suffix. |
| `decode_tok_s` | Native decoded-token count / native decode seconds theo semantic của phiên bản. Lưu cách engine tính first token/EOS; không tự trừ 1 nếu chưa xác minh. |
| `e2e_request_ms` | Từ trước HTTP send đến terminal event hợp lệ, bao gồm truyền dữ liệu. Không suy bằng tổng các timing engine. |
| `cold_activation_to_response_ms` | Từ activation ban đầu đến terminal response, bao gồm readiness/probe gap; lưu timeline thực tế thay vì cộng tùy ý. |
| Idle VRAM | Median device used memory trong 2 giây trước activation, model chưa resident. |
| Loaded VRAM | Median device used memory trong 2 giây sau readiness và trước generation; ghi rõ engine có cấp phát KV/buffer ngay ở thời điểm này hay chưa. |
| Peak VRAM | Maximum sample trong load, request và toàn trial; ba scope riêng. Lưu absolute bytes và delta so với idle. Đây là sampled peak, có thể bỏ lỡ đỉnh giữa hai mẫu. |

Ollama trả duration ở nanoseconds; final streaming response chứa usage, gồm total prompt count, cached count và duration cho uncached tokens ở API hiện tại. Adapter phải kiểm tra bản pin có đúng fields/semantics này. [Ollama usage](https://docs.ollama.com/api/usage).

llama-server expose native timings và các điều khiển cache/sampling; flags và semantics phải được đối chiếu lại tại commit pin. [llama-server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

Không chia cho zero; duration/token count thiếu hoặc bất hợp lý tạo null + reason. Empty output không có TTFT, nhưng có thể có valid E2E. Không suy decode speed từ khoảng cách giữa chunks.

### 2.5 Cold, warm, cache và repetitions

Đơn vị phân tích: `experiment × configuration × prompt_id × run_mode`. Không trộn cold và warm, các prompt lengths, cache policy hoặc deployment mode vào một trung bình.

**Cold mode = model-cold, OS-page-cache không kiểm soát.** Trước mỗi trial: unload/stop owned model process → xác nhận model absent và VRAM về baseline trong tolerance đã pin → lấy idle → activate/load → lấy loaded → chạy request đầu tiên → unload. Không inference warmup vào model giữa activation và request. Preload phải không generate; nếu backend có internal startup warmup, lưu hành vi đó. Daemon Ollama có thể sống sẵn; llama.cpp dùng dedicated process mỗi lần. Sự bất đối xứng lifecycle này phải hiện trong report.

**Warm mode = model resident, inference đã warmup.** Sau load, chạy 2 warmup requests không đưa vào aggregates; trước mỗi request đo chuẩn hóa cache như dưới đây. Warm request không tính load time vào TTFT/E2E. Load observation thuộc session và không được nhân bản thành nhiều mẫu độc lập.

**Cache là điều kiện nghiệm thu, không phải giả định.** Ollama không được mặc định có cùng cache-disable API như llama.cpp. MVP dùng một sequence và một cache-eviction request trước mỗi warm measurement, áp dụng cho cả hai engine. Eviction prompt là raw text có first substantive token khác prompt đo, dài ít nhất bằng prompt đo; generate tối đa 1 token. Giữ cơ chế prefix reuse của Ollama, đặt `cache_prompt=true` ở llama.cpp, rồi kiểm tra eviction thực sự loại bỏ prefix cần tránh; lưu eviction event ngoài measurement window. Eviction requests dùng cùng bytes/settings giữa các engine khi so E2, và có budget riêng không nhập output-token statistics.

Sau eviction, yêu cầu measured prompt không reuse substantive prefix; chỉ chấp nhận BOS cố định nếu kiến trúc bắt buộc, được ghi và khớp giữa hai engine. Xác minh bằng native cache counts/logs trên các cặp prompt lặp lại. Eviction không tự nó chứng minh cache đã sạch. Nếu không chứng minh được, warm prefill comparison bị chặn; chọn version/toolchain có telemetry đủ hoặc một cơ chế clear-cache được xác minh rồi chạy lại preflight. Không thay warm bằng restart và giữ nguyên nhãn.

Eviction có thể ảnh hưởng nhiệt độ; chạy giống nhau ở các engine, có settle interval cố định 1 giây và thu temperature/clocks. Không bao gồm eviction trong peak request VRAM hoặc latency; vẫn lưu peak toàn session riêng.

Protocol chính thức:

- Performance suite: 6 prompts cố định, 2 prompts mỗi nhóm độ dài khoảng 128/512/1024 token trên reference tokenizer; lưu actual counts cho từng model.
- Mỗi cell/prompt: 5 cold trials và 10 warm trials, chia warm thành 2 sessions × 5 trials; mỗi session có 2 warmups. Cache eviction trước từng warm trial.
- Balanced randomized blocks: mỗi block chứa đủ configurations, thứ tự shuffle bằng `schedule_seed=20261002`; lưu schedule trước khi chạy. Không chạy hết engine A rồi mới bắt đầu toàn bộ B.
- Cold blocks và warm blocks riêng; không chạy GPU song song. Có cooldown policy chung và timeout; không đổi policy giữa campaign.
- Defaults cho control plane: connect timeout 5 giây, activation 180 giây, first-content 120 giây, toàn request 600 giây, unload 60 giây. Trước mỗi block, chờ tối đa 180 giây để median GPU utilization trong 2 giây ≤10% và temperature ≤70°C; nếu không đạt thì ghi blocked-environment và dừng campaign, không thu thử rồi chọn các trial nhanh. Pilot có thể điều chỉnh các ngưỡng trước freeze; điều kiện sau khi request bắt đầu được ghi thành telemetry, không dùng để loại outlier tùy ý.
- Quality: 60 items cố định, mỗi category 20; 1 response/item/config với greedy profile. Không lấy best-of-repetitions. Chấm ngoài timed measurement.
- Quality suite có version/hash, answer key tách khỏi prompt; không tuyên bố đại diện cho toàn bộ năng lực LLM hay không bị training contamination.
- Pilot chọn khả thi về memory và kiểm tra protocol, không dùng quality score để chọn model thắng hoặc chỉnh prompt. Freeze mọi cấu hình trước official run.

## 3. Hợp đồng module và dữ liệu

`Backend` là Python Protocol đơn giản:

- `inspect()` → version, capabilities, loaded/offload information.
- `load(spec)` → `LoadObservation`.
- `stream(request)` → iterator `StreamEvent`; không ghép toàn bộ response trước khi phát event.
- `prepare_cache(request)` → `CachePreparation`; runner áp policy và kiểm tra evidence.
- `unload()` / `close()` → idempotent cleanup, chỉ quản lý process/artifact do harness sở hữu.

`StreamEvent`: local monotonic receipt time, kind (`text`, `thinking`, `final`, `error`), payload, native usage khi có. Với model non-thinking, thinking bất ngờ là protocol violation, vẫn lưu output gốc.

`RunRecord` bắt buộc có:

- `schema_version`, campaign/trial/attempt/session/block IDs, phase, experiment, run mode, synthetic flag.
- Source model revision; GGUF SHA-256/quant; engine identity; environment/config/workload/template hashes.
- Requested và effective settings; prompt hash/path; start/end clocks; raw stream/log/monitor paths.
- Load observations hoặc reference đến load event; client timings; native timings và provenance; token counts; cache evidence; VRAM metrics/scope.
- Stop reason, output text/hash, status (`ok`, `oom`, `timeout`, `interrupted`, `invalid`, `unsupported`, `infeasible`), error và exclusion reasons.

Raw layout: campaign manifest + resolved config + environment + schedule + append-only `runs.jsonl`; mỗi attempt có stream events, output, engine log và GPU trace riêng. Artifact lớn GGUF để ngoài Git, manifest/hash phải nằm trong repo. Mỗi lần resume giữ nguyên trial ID nhưng tạo attempt ID mới; không ghi đè lỗi cũ. Analysis chọn attempt hợp lệ hoàn tất đầu tiên theo thứ tự thời gian của trial, không theo tốc độ hoặc quality; trial đã có attempt hợp lệ không chạy lại khi resume. Báo cáo riêng initial-attempt failure rate và unresolved-trial failure rate để không che reliability bằng retries.

## 4. Tasks MVP — thực hiện theo thứ tự ID

Mỗi task kết thúc bằng files hoàn chỉnh, tests tương ứng và một ghi chú validation ngắn. Không chuyển sang official GPU run khi gate trước đó chưa đạt. “CPU” bao gồm unit/mock; “GPU thật” nghĩa là đúng RTX 3050 mục tiêu.

### T01 — Khởi tạo Python project

- **Mục tiêu:** có package cài được và CLI thống nhất.
- **Files:** `pyproject.toml`, `uv.lock`, `.gitignore`, `src/quant_benchmark_lab/__init__.py`, `cli.py`, `tests/conftest.py`, các thư mục đích.
- **Implementation:** cấu hình Python ≥3.11, entry point `qbl`, dependency groups runtime/dev/prepare, pytest markers `integration` và `gpu`; dùng Ruff cho lint/format. Ignore weights, build directories, cache và result tạm.
- **Acceptance:** `uv sync --frozen`, `uv run qbl --help`, import package và Ruff đều thành công; chưa chạy tải model/GPU tự động.
- **Dependencies:** không.
- **Phần cứng:** CPU.
- **Tests:** smoke test CLI help và package entry point.

### T02 — Ghi protocol và preregistration

- **Mục tiêu:** biến mục 2 thành spec có version, không để agent tự chọn cách đo.
- **Files:** `docs/methodology.md`, `docs/limitations.md`, `configs/protocol.yaml`.
- **Implementation:** ghi metric boundaries, cold/warm state transitions, cache acceptance, repetitions, model-selection rule, exclusion policy, sampling defaults. Đánh dấu config `draft` cho đến T21.
- **Acceptance:** mỗi metric có unit, source, start/end boundary và missing-data rule; mọi thay đổi sau freeze yêu cầu campaign mới.
- **Dependencies:** T01.
- **Phần cứng:** CPU.
- **Tests:** kiểm tra các keys protocol bắt buộc sau khi T03 có validator; review checklist không có định nghĩa metric mâu thuẫn.

### T03 — Config và ResultSchema

- **Mục tiêu:** reject cấu hình mơ hồ trước khi gọi engine.
- **Files:** `config.py`, `schema.py`, `configs/runtime.yaml`, `configs/models.yaml`, `configs/experiments/*.yaml`, `tests/unit/test_config.py`, `test_schema.py`.
- **Implementation:** Pydantic strict models, reject unknown keys, resolved config có mọi giá trị; canonical JSON hash; typed settings cho workload/model/backend/protocol; schema metric nullable với reason. Đăng ký `qbl validate`.
- **Acceptance:** E1 chỉ đổi quant; E2 chỉ đổi engine và giữ chung GGUF/settings/prompt; E3 chỉ đổi model-derived fields. Invalid context, repetitions, hash hoặc unit phải fail trước run.
- **Dependencies:** T01–T02.
- **Phần cứng:** CPU.
- **Tests:** round-trip serialization, config hashing ổn định, unknown field, invariant violation, negative durations và missing reason.

### T04 — Capture environment và doctor

- **Mục tiêu:** lưu đủ thông tin để giải thích khác biệt giữa các lần chạy.
- **Files:** `environment.py`, cập nhật `cli.py`, `tests/unit/test_environment.py`.
- **Implementation:** `qbl doctor` lấy CPU model/cores, RAM, OS/kernel, Python/package versions, GPU name/UUID/total VRAM, driver, installed CUDA toolkit và runtime/build CUDA nếu quan sát được, engine versions/binary hashes, llama.cpp commit/build flags, repo commit/dirty state, power mode và process GPU hiện tại. Không gọi nhãn CUDA trên `nvidia-smi` là installed toolkit.
- **Acceptance:** missing tool/GPU tạo explicit unavailable; official GPU mode thiếu dữ liệu thiết yếu phải fail. Không dump toàn bộ environment variables chứa secret.
- **Dependencies:** T03.
- **Phần cứng:** CPU/mock; xác nhận GPU fields ở T21.
- **Tests:** mocked command output, CUDA absent, non-Git workspace, unknown power field, command timeout.

### T05 — Pin và build toolchain

- **Mục tiêu:** cùng binary và conversion tool trong một campaign.
- **Files:** `scripts/bootstrap_llama.sh`, `locks/toolchain.json`, `docs/reproduction.md`.
- **Implementation:** chọn full commit SHA llama.cpp hỗ trợ các architecture; build `llama-server`, `llama-quantize` bằng CMake Release + CUDA; lưu compiler/CMake/CUDA/flags, binary hashes và `--help`. Pin Ollama version/binary hash và Python preparation dependencies. Chỉ clone/build trong thư mục chỉ định.
- **Acceptance:** lock không dùng `latest`/floating branch; version mismatch chặn official run; conversion và quantization cùng toolchain provenance. Commands tham chiếu README của commit đã pin.
- **Dependencies:** T04.
- **Phần cứng:** CPU để chuẩn bị; CUDA toolkit để build GPU target, runtime CUDA xác nhận ở T21.
- **Tests:** mock download/build failure, checksum mismatch, rerun idempotent; CPU binary smoke test nếu có.

### T06 — Chuẩn bị GGUF và artifact manifest

- **Mục tiêu:** chứng minh các quant cùng lineage, các engine dùng cùng bytes.
- **Files:** `artifacts.py`, `scripts/prepare_models.py`, `locks/models.json`, `tests/unit/test_artifacts.py`.
- **Implementation:** tải source theo revision SHA; lưu license; convert F16 bằng converter của llama.cpp; tạo Q8_0 và Q4_K_M trực tiếp từ F16 bằng `llama-quantize`; lưu command, logs, tensor metadata, sizes, SHA-256 và source tokenizer/template hashes. Không dùng importance matrix ở MVP. Hỗ trợ dry-run và kiểm tra disk/RAM trước conversion.
- **Acceptance:** manifest truy được mọi GGUF về source+toolchain; file đổi 1 byte bị phát hiện; không ghi đè artifact đã có checksum khác. Import Ollama ở task lifecycle sau phải dùng đường dẫn GGUF này.
- **Dependencies:** T03, T05.
- **Phần cứng:** CPU, đủ RAM/disk; không cần GPU để quantize.
- **Tests:** fake converter/quantizer, failed subprocess, partial artifact, lineage sai, checksum đúng/sai.

### T07 — Fixed workloads và prompt renderer

- **Mục tiêu:** khóa dữ liệu và prompt trước khi chạy.
- **Files:** `prompts.py`, `quality/dataset.py`, `data/performance/prompts.jsonl`, `data/quality/{items,answers}.jsonl`, `locks/workloads.json`, `tests/unit/test_prompts.py`.
- **Implementation:** 6 performance prompts; 60 quality items theo mục 2.5; renderer theo template/tokenizer revision, có explicit thinking/BOS policy. Mỗi item có stable ID/category/scorer/output rule. Tạo cache-eviction prompts tách khỏi suite chấm điểm. License/provenance đi cùng dataset.
- **Acceptance:** cùng input/config sinh cùng bytes/hash; answer key không có trong request; mọi rendered prompt vừa context; không sửa prompts theo kết quả model.
- **Dependencies:** T03, T06.
- **Phần cứng:** CPU.
- **Tests:** golden prompt bytes, Unicode, special tokens, generation suffix, non-thinking template, context overflow và answer leakage.

### T08 — Backend contract và streaming primitives

- **Mục tiêu:** runner dùng cùng interface cho cả hai engine.
- **Files:** `backends/base.py`, `backends/fake.py`, `backends/streaming.py`, `metrics.py`, `tests/contract/test_backend.py`, `tests/unit/test_streaming.py`.
- **Implementation:** Protocol và event types ở mục 3; HTTP timeout rõ; injectable monotonic clock, fake scripted backend. Parser NDJSON/SSE chịu được frame chia qua nhiều TCP chunks, UTF-8 split và heartbeat. Chỉ timestamp khi event được client nhận/giải mã đủ.
- **Acceptance:** TTFT bắt đầu ở first nonempty generated text, terminal event chỉ một lần; không phụ thuộc chunk count; backend error không trở thành thành công.
- **Dependencies:** T03.
- **Phần cứng:** CPU/mock.
- **Tests:** fake clock kiểm tra công thức chính xác, split frames, empty/whitespace output, multiple frames/chunk, truncated stream, duplicate final và disconnect.

### T09 — Ollama lifecycle và import

- **Mục tiêu:** load/unload GGUF có kiểm soát, không đụng phiên Ollama của người dùng.
- **Files:** `backends/ollama.py`, `tests/unit/test_ollama_lifecycle.py`.
- **Implementation:** managed daemon ở port/model directory riêng; giới hạn một model và một parallel sequence; tạo Modelfile `FROM` local GGUF, import và xác minh model blob digest. Explicit preload không generation, `keep_alive` đủ dài, explicit unload, poll model absence. Thu native load duration và wall activation time riêng.
- **Acceptance:** import không requantize; loaded artifact hash đúng; repeat load/unload/close an toàn; không kill process không thuộc harness; lỗi load có log.
- **Dependencies:** T04, T06, T08.
- **Phần cứng:** CPU/mock; CPU integration có thể chạy model nhỏ.
- **Tests:** mock endpoints/process, wrong blob, preload generates unexpectedly, unload timeout, daemon exit, repeated cleanup.

### T10 — Ollama streaming và native metrics

- **Mục tiêu:** đo request mà không mất dữ liệu streaming hoặc usage.
- **Files:** cập nhật `backends/ollama.py`, `metrics.py`, `tests/fixtures/ollama/`, `tests/unit/test_ollama_stream.py`.
- **Implementation:** raw prompt, explicit options, streaming; adapter ns → normalized unit; lấy usage final, cached tokens và stop reason theo version pin; lưu payload gốc. Phân biệt output text/thinking. Kiểm tra option support qua source/docs/probes, không coi HTTP 200 là bằng chứng option có hiệu lực.
- **Acceptance:** metrics đúng với fixture đã biết; missing final khiến trial invalid; không derive input evaluation từ tổng prompt khi thiếu cache accounting.
- **Dependencies:** T07–T09.
- **Phần cứng:** CPU/mock; sample API thật CPU ở T13/T22.
- **Tests:** cached/full prompt examples, zero duration, error stream, unexpected thinking, ignored setting, early EOS và count mismatch.

### T11 — llama-server lifecycle

- **Mục tiêu:** quản lý một server/slot/model có launch configuration rõ.
- **Files:** `backends/llamacpp.py`, `tests/unit/test_llamacpp_lifecycle.py`.
- **Implementation:** launch binary pin bằng argument list, context/thread/batch/KV/offload/flash-attention flags explicit, một slot, log stdout/stderr; health polling không inference; lưu launch command, PID và readiness time. Parse load/offload evidence theo commit pin; shutdown có timeout và cleanup owned children.
- **Acceptance:** health timeout/server crash có record; model_load_wall bao gồm spawn→ready; native load field thiếu phải null, không dựng từ request TTFT.
- **Dependencies:** T04–T06, T08.
- **Phần cứng:** CPU/mock; GPU flags xác nhận T21.
- **Tests:** process fake, unknown flag, readiness race, port conflict, log parser và termination timeout.

### T12 — llama-server streaming và native metrics

- **Mục tiêu:** cùng semantics ở mức harness với T10.
- **Files:** cập nhật `backends/llamacpp.py`, `metrics.py`, `tests/fixtures/llamacpp/`, `tests/unit/test_llamacpp_stream.py`.
- **Implementation:** native `/completion` SSE, explicit sampling/stop/cache options; lấy timings/counts từ final event, ms normalization; map `n_predict` từ `max_tokens`. Lưu cached/processed tokens nếu endpoint/log có. Kiểm tra tokenizer endpoint dùng đúng GGUF.
- **Acceptance:** cùng contract suite T08 pass; giữ original native counts và timing definitions; không giả định OpenAI-compatible usage có đủ timing.
- **Dependencies:** T07–T08, T11.
- **Phần cứng:** CPU/mock.
- **Tests:** fragmented SSE, ping, native timing conversion, stop vs length, zero output, malformed/missing final và unsupported request option.

### T13 — Equivalence gate và cache validation

- **Mục tiêu:** chặn so sánh E2 khi đầu vào/thực thi không tương đương.
- **Files:** `protocol.py`, cập nhật `prompts.py`, hai backends, `tests/contract/test_equivalence.py`, `docs/methodology.md`.
- **Implementation:** tạo `equivalence.json` so SHA GGUF, raw bytes, stop/thinking/sampling/context, token counts/BOS, effective runtime. Implement cache preparation ở mục 2.5; probe repeated prompt trước/sau eviction, cả short/long prompts. Capabilities thiếu dẫn đến fail có lý do; lưu probe evidence.
- **Acceptance:** input/settings lệch bị chặn; sau eviction không còn substantive prefix cache; hai engine có residual BOS cache khớp. Không yêu cầu generated text identical để pass. GPU offload equality chỉ hoàn tất T21.
- **Dependencies:** T10, T12.
- **Phần cứng:** CPU/mock và real CPU backend; GPU xác nhận lại T21.
- **Tests:** intentional template/BOS/quant mismatch, cache hit bị phát hiện, cache unknown bị chặn, ignored option, valid pair và output khác nhưng inputs hợp lệ.

### T14 — GPUMonitor và phase windows

- **Mục tiêu:** đo VRAM có timeline gắn đúng trial.
- **Files:** `monitoring/gpu.py`, cập nhật `schema.py`, `tests/unit/test_gpu_monitor.py`.
- **Implementation:** NVML qua `nvidia-ml-py`, sample interval mặc định 20 ms; worker thread riêng; timestamp monotonic; device used/free, process VRAM nếu hỗ trợ, temperature, clocks, utilization, power/throttle reason nếu có. Marker idle/load/loaded/eviction/request/cleanup. Buffer samples rồi flush ngoài timed path.
- **Acceptance:** xuất full samples + actual intervals/missed samples; idle/loaded/peak theo đúng scope; device và process memory tách; unsupported process accounting có reason. Timeout cleanup không để thread chạy nền.
- **Dependencies:** T03–T04, T08.
- **Phần cứng:** mock để phát triển; GPU thật để nghiệm thu ở T21.
- **Tests:** synthetic memory trace có peak ở load và request khác nhau, phase boundaries, NVML absent, dropped samples, monitor thread exception.

### T15 — Raw result storage và integrity

- **Mục tiêu:** giữ chứng cứ kể cả khi benchmark bị dừng.
- **Files:** `storage.py`, `tests/unit/test_storage.py`, kết cấu `results/raw/`.
- **Implementation:** immutable campaign manifest; append-only run records; attempt-specific event/log files; atomic write cho metadata; checksum/index các artifact; giữ partial trace khi lỗi. Không disk flush blocking từng token trong measurement.
- **Acceptance:** campaign đọc lại không mất settings/native payload; record chưa complete không được aggregate; không overwrite attempts hoặc official data bằng fixture.
- **Dependencies:** T03, T08, T14.
- **Phần cứng:** CPU/mock.
- **Tests:** simulated crash, partial JSONL tail, duplicate ID, checksum mismatch, permission error và synthetic isolation.

### T16 — Scheduler và BenchmarkRunner

- **Mục tiêu:** chạy chính xác protocol bằng một orchestration loop đơn giản.
- **Files:** `runner.py`, cập nhật `protocol.py`, `cli.py`, `tests/unit/test_runner.py`.
- **Implementation:** `qbl plan` xuất deterministic balanced schedule và số trials; `qbl run` thực hiện cold/warm state machine, warmup/cache prep/settle, monitor markers, HTTP timing, storage. Assert one active request/model. Thời gian preprocessing, render, scoring và report ngoài request timer.
- **Acceptance:** với fake backend, đúng 5 cold + 10 warm/prompt/config, warm chia 2 sessions; 2 warmups/session được lưu nhưng excluded; load event mỗi session không bị nhân thành observations độc lập. Có `--dry-run` không gọi engine.
- **Dependencies:** T13–T15.
- **Phần cứng:** CPU/mock.
- **Tests:** exact call order, correct counts, fixed seed schedule, warmup exclusion, no overlapping requests, phase timing và lifecycle cleanup.

### T17 — Failure handling và resume

- **Mục tiêu:** không tạo survivor bias hoặc mất dữ liệu khi lỗi.
- **Files:** cập nhật `runner.py`, `storage.py`, `cli.py`, `tests/unit/test_resume.py`.
- **Implementation:** classify OOM/timeout/interruption/infeasible/invalid; không tự retry request đo. `qbl run --resume` yêu cầu manifest/config/artifact/environment identity phù hợp, tạo attempt mới; warm session interrupted phải warmup lại. Environment thay đổi đáng kể tạo campaign mới.
- **Acceptance:** failed trials còn nguyên trong report; không đổi settings sau OOM; có thể dừng/resume mà không double-count completed trials. Exclusion deterministic, không bỏ statistical outliers sau khi thấy kết quả.
- **Dependencies:** T16.
- **Phần cứng:** CPU/mock.
- **Tests:** SIGINT simulation, timeout, OOM, failed unload, interrupted session, changed hash và duplicate attempt selection.

### T18 — Objective QualityEvaluator

- **Mục tiêu:** có quality score kiểm toán được, chạy offline.
- **Files:** `quality/scorers.py`, `quality/evaluator.py`, cập nhật `cli.py`, `tests/unit/test_quality.py`.
- **Implementation:** exact match với normalization khai báo từng task; numeric parser theo output contract, absolute/relative tolerance per item; JSON extraction chấm exact fields/value. Invalid format, ambiguous answer và truncated output có status riêng. `qbl evaluate` lưu answer, normalized prediction, score và scorer version.
- **Acceptance:** 60 items có score/explanation deterministic; mỗi item tối đa 1 điểm; report category accuracy và macro-average. Model failures tính 0 trong completion-inclusive score và còn có answered-only score để phân biệt hạ tầng/năng lực; không chọn câu trả lời tốt nhất từ attempts.
- **Dependencies:** T07, T15–T17.
- **Phần cứng:** CPU cho scoring; generation thật cần backend, GPU cho campaign.
- **Tests:** units, decimals, sign, scientific notation, tolerance boundary, malformed JSON, extra fields, empty/truncated output và normalization không xóa mất đáp án sai.

### T19 — Aggregation và comparison rules

- **Mục tiêu:** thống kê đúng cấp độ, không gộp các điều kiện khác nhau.
- **Files:** `reporting/aggregate.py`, `tests/unit/test_aggregate.py`.
- **Implementation:** group theo mục 2.5 + workload/settings/cache/offload identity; mean, median, sample std (`ddof=1`), n planned/attempted/valid/failed/excluded và missing per metric. Tính tok/s từng trial rồi aggregate; pooled total tokens/total seconds nếu có là field riêng. Pair engine results theo block/prompt/repetition khi đủ cả hai.
- **Acceptance:** n=1 → std null; no valid samples → metrics null; không biến missing thành zero. Load stats dùng unique load events. Không lấy mean tok/s qua model tokenizers để gọi là cùng lượng công việc. Quality n là số items, không phải performance repetitions.
- **Dependencies:** T15, T18.
- **Phần cứng:** CPU.
- **Tests:** hand-calculated fixtures, unequal group lengths, missing metrics, invalid-only group, duplicate load refs, unpaired trials và rates-vs-pooled distinction.

### T20 — Reporter và portfolio figures

- **Mục tiêu:** từ raw data tạo được báo cáo độc lập, không chạy lại inference.
- **Files:** `reporting/reporter.py`, `reporting/plots.py`, cập nhật `cli.py`, `tests/unit/test_reporter.py`.
- **Implementation:** `qbl report` xuất CSV/Markdown/PNG: separate E1/E2/E3, cold/warm, load/TTFT/prefill/decode/E2E/VRAM; quality vs speed và quality vs VRAM; actual output lengths và failures hiển thị cùng kết quả. Link mỗi bảng về campaign/config/raw provenance. Gắn synthetic watermark lên toàn bộ fixture reports.
- **Acceptance:** report không claim winner nếu equivalence gate fail; ô thiếu là N/A kèm reason; charts có units/n/error-bar definition. Near-limit claim dựa peak đo thật. Narrative chỉ lấy từ records, không tự sinh số hoặc suy nhân quả vượt experimental design.
- **Dependencies:** T19.
- **Phần cứng:** CPU.
- **Tests:** deterministic report from fixed fixture, missing series, mixed synthetic/real rejection, invalid comparisons, links và plot labels; visual review report mẫu.

### T21 — Pilot trên RTX 3050 và freeze campaign

- **Mục tiêu:** chứng minh assumptions quan trọng trên phần cứng đích trước thu số chính thức.
- **Files:** `cli.py` cho `qbl preflight`, cập nhật draft configs/locks, `results/raw/pilot-*/`, `docs/methodology.md` phần frozen decisions.
- **Implementation:** chạy mọi matrix cell với shortest/longest prompts và max output budget; xác nhận full offload, peak/headroom, token parity, cache eviction, actual settings, stop/thinking behavior. Thử monitor on/off theo cặp 5 repetitions ở model nhỏ: nếu median E2E overhead >3%, điều tra rồi tăng interval chung và pilot lại, ghi giới hạn lấy mẫu. Ghi display workload, power profile và temperature; pin baseline tolerance 64 MiB cho clean unload trừ khi pilot chứng minh cần threshold khác.
- **Acceptance:** có E1 F16/Q8/Q4 trên ít nhất model 0.5B, E2 pair valid, E3 có ít nhất 3 model và một near-limit model đo thật. Một cell dự kiến optional infeasible được giữ có lý do. Cache/offload chưa xác minh thì gate fail. Freeze resolved config, hashes, schedule, thresholds; pilot records không nhập official aggregates.
- **Dependencies:** T05–T20.
- **Phần cứng:** bắt buộc RTX 3050 4GB thật.
- **Tests:** hardware preflight checklist + GPU integration tests cho offload/OOM/unload/VRAM attribution; kiểm tra telemetry completeness và monitor overhead.

### T22 — CI và end-to-end reproducibility checks

- **Mục tiêu:** thay đổi harness không phá methodology đã kiểm tra.
- **Files:** `.github/workflows/ci.yml`, `tests/integration/test_fake_campaign.py`, `test_cpu_backends.py`, `test_gpu_preflight.py`.
- **Implementation:** default CI chạy lint/unit/contract/fake campaign từ plan→run→evaluate→report; real CPU integration chạy opt-in với tiny local GGUF và pinned engines; GPU tests có marker, chạy trên máy lab. Fixtures ghi version và nguồn, không refresh tự động theo latest.
- **Acceptance:** mock campaign chạy offline; CI không tải model lớn; real CPU smoke test cả hai backend thành công; GPU gate T21 có bằng chứng. Chỉ yêu cầu coverage các ranh giới đo và error paths, không đặt phần trăm coverage hình thức.
- **Dependencies:** T20–T21.
- **Phần cứng:** CPU cho CI; GPU cho marked suite.
- **Tests:** toàn bộ suite phù hợp; report regeneration từ raw bất biến; two identical fake campaigns chỉ khác metadata thời gian cho phép.

### T23 — Chạy official experiments và quality suite

- **Mục tiêu:** tạo kết quả thật cho cả ba experiments theo frozen protocol.
- **Files:** `results/raw/<official_campaign>/`, `reports/<official_campaign>/`, campaign manifest/index.
- **Implementation:** doctor → checksum/config validation → thực thi schedule đã freeze → quality generation riêng → offline evaluate → aggregate/report. Không update engine/model/dependency giữa campaign. Kiểm tra pending/failed trial IDs và resume theo T17; mọi exclusions theo preregistered rules.
- **Acceptance:** đủ planned trials hoặc có explicit failures; không có unexplained gaps; baseline comparison chỉ dùng qualified cells; outputs, native usage, GPU traces, environment và logs truy xuất được. Không bổ sung số bằng estimate vào metric thiếu.
- **Dependencies:** T22.
- **Phần cứng:** bắt buộc GPU thật; evaluate/report dùng CPU.
- **Tests:** final data audit về completeness/hashes/schema, measurement validity, offload/cache flags, synthetic contamination và duplicated trials; không chọn rerun chỉ vì kết quả chậm.

### T24 — Hoàn thiện hướng dẫn reproduce và portfolio

- **Mục tiêu:** người khác chạy lại được và hiểu giới hạn kết luận.
- **Files:** `README.md`, `docs/reproduction.md`, `docs/limitations.md`, report index; cập nhật `AGENTS.md` khi project đã có structure thực tế và task triển khai cho phép.
- **Implementation:** hướng dẫn dependency/toolchain/artifact prep, CLI sequence, hardware profile, thời gian/disk thực tế từ campaign; small downloadable raw sample kèm checksums và nơi lưu full artifacts. Viết methodology, findings chỉ dựa T23, limitations và 5–8 câu hỏi interview về cache, TTFT, cold start, offload, tokenization, replication và quality validity.
- **Acceptance:** từ môi trường sạch, agent khác chạy được fake campaign; từ raw được công bố, tái tạo được tables/figures không cần GPU; hướng dẫn official GPU run đầy đủ. Không claim reproduce cùng số tuyệt đối trên GPU khác.
- **Dependencies:** T23.
- **Phần cứng:** CPU để kiểm tra tài liệu/report; GPU nếu thực hiện lại campaign.
- **Tests:** walkthrough README trên clean environment, resolve artifact links/checksums và regenerate report so với manifest.

## 5. Improvements sau MVP

Các task dưới đây không chặn T24. Không đưa vào main experiment trước khi cập nhật protocol/version và chạy campaign mới nếu thay đổi phép đo.

| ID | Mục tiêu và files | Implementation details | Acceptance và tests | Dependencies | GPU |
|---|---|---|---|---|---|
| I01 | Coding quality; `quality/code_runner.py`, coding dataset | Rootless container, network off, CPU/RAM/PID/time limits, read-only inputs; pin test suite; không chạy arbitrary model code bằng subprocess trực tiếp trên host | pass@1 theo fixed tests; tests cho timeout, endless output, denied network và sandbox failure; fail closed nếu sandbox thiếu | T18, T24 | CPU chấm; GPU generate |
| I02 | LLM judge proxy; `quality/judge.py`, rubrics/config | Fixed rubric/judge revision/settings; anonymous answer IDs; randomized order seed; pairwise position swap; tách mapping identity khỏi judge input; lưu raw judgments | Label rõ proxy; tests cho identity leakage, order reproducibility, parser failure; audit một sample bằng người | T18, T24 | Tùy judge; không chạy cạnh inference benchmark |
| I03 | Uncertainty và nhiều sessions; reporting stats | Paired/block bootstrap trên prompts/sessions phù hợp; ngày chạy khác nhau; preregister sample size; không bootstrap tokens như observations độc lập | CI/repeated-session plots có sampling unit; tests từ synthetic distributions và missing pairs | T19, T24 | GPU để thêm observations |
| I04 | Hybrid CPU/GPU offload; config/schema/report | Sweep fixed layer counts, threads, RAM usage, giữ quant/context; separate experiment ID và deployment mode | Không gộp với full-GPU; tests cho grouping/offload detection; đo thật memory và speed | T21, T24 | GPU thật |
| I05 | Cache-hit và context-length experiments | Warm full-prefix cache riêng; context/input sweeps độc lập; xác nhận cached tokens và allocation | Cache-hit không nhập uncached prefill chart; tests cho strata và throughput denominator | T13, T24 | GPU thật |
| I06 | Disk-cold startup | Protocol OS page-cache kiểm soát riêng, dedicated environment, permissions/commands ghi rõ; tách daemon/process/kernel warmness | Không gọi model-cold là disk-cold; tests lifecycle/mode labeling; đo repeatable trên lab | T21, T24 | Máy lab thật |
| I07 | Energy và hiệu quả | Tích phân sampled power theo time, idle-adjusted energy báo riêng, quality/J và tokens/J đúng scope | Ghi uncertainty từ sampling; tests integration trên traces tổng hợp và sensor missing | T14, T24 | GPU thật có telemetry |

## 6. Thứ tự triển khai và các gate

Thực hiện tuần tự `T01 → T02 → … → T24`. Các dependencies đã liệt kê cho phép biết task nào thực sự bị chặn; không cần framework điều phối agent.

1. **Gate A — spec/data contract:** T01–T08. Config, schema, prompts, metric definitions và fake streaming có tests.
2. **Gate B — backend correctness:** T09–T15. Hai backend, equivalence/cache evidence, monitor và raw storage hoàn chỉnh.
3. **Gate C — mock campaign:** T16–T20. Chạy trọn pipeline bằng fake data, report mang synthetic label.
4. **Gate D — hardware validity:** T21–T22. Matrix fit GPU, near-limit verified, settings/cache/offload đạt, configs frozen.
5. **Gate E — portfolio-ready:** T23–T24. Có official raw data, quality scores, reports và reproduction guide.

Các lệnh đích cần được implement, chưa có sẵn ở thời điểm viết kế hoạch:

```text
uv sync --frozen
uv run qbl doctor
uv run qbl validate --config configs/experiments/engine.yaml
uv run qbl preflight --config configs/experiments/engine.yaml
uv run qbl plan --config configs/experiments/engine.yaml --output <schedule>
uv run qbl run --schedule <schedule>
uv run qbl evaluate --campaign <campaign_id>
uv run qbl report --campaign <campaign_id>
uv run pytest -m "not gpu and not integration"
```

## 7. Definition of Done cho MVP

- Ba experiments được giữ riêng; mọi headline comparison có evidence cho các biến kiểm soát.
- Có model nhỏ và model gần giới hạn VRAM được xác minh; quant không fit được công bố là infeasible, không giả lập kết quả.
- Các metric yêu cầu có raw evidence, unit và measurement boundary; missing data hiển thị rõ. Native load timer có thể thiếu nhưng operational load-to-ready bắt buộc đo riêng.
- Cold/warm, warmups, cache preparation, actual token counts và offload mode đều truy xuất được theo trial.
- Kết quả official không chứa fixture/pilot data; mean/median/std và n đi cùng failures/exclusions.
- Quality objective có answer keys/scorer version; không có LLM judge score được gọi là ground truth.
- Reporter tái tạo được từ raw results mà không load model. Artifact/toolchain/config/environment lineage đủ để thực hiện lại campaign.
- README giải thích được giới hạn của một GPU, một workload suite, tokenizers khác nhau, streaming buffering và sampled VRAM peaks.

## 8. Nguồn chính thức để agent đối chiếu khi pin phiên bản

- [Ollama Generate API](https://docs.ollama.com/api/generate): raw/stream/keep-alive/thinking và response fields.
- [Ollama usage](https://docs.ollama.com/api/usage): duration units, total/cached input và final streaming usage.
- [Ollama Modelfile](https://docs.ollama.com/modelfile): import reference, sampling và stop parameters.
- [Ollama import](https://docs.ollama.com/import): import GGUF đã chuẩn bị.
- [Ollama source API types](https://github.com/ollama/ollama/blob/main/api/types.go): đối chiếu supported fields tại version pin.
- [llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md): native completion, timings, cache và runtime flags.
- [llama-quantize](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md): tool tạo GGUF quant.

Các URL tài liệu có thể thay đổi; T05 phải lưu commit/version và source references tương ứng với binary thực sự chạy. Không dùng tài liệu của `master` để ngầm bảo đảm hành vi của một binary cũ.
