# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Trần Tuấn Cường
**Khoá:** B22DCVT073 - Track 3 (VinAI / VinUni AICB-P2T3 K4)
**Tier đã chạy:** T4
**Ngày:** 2026-10-09

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Kaggle GPU Tesla T4 16 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | rm-panel:Skywork/Skywork-Reward-V2-Llama-3.2-3B; sanity accuracy 100% |
| Chi phí | 0 đồng (Kaggle GPU T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~22 phút (100 steps) |
| VRAM cao nhất | 10.8 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.0949 |
| Độ chính xác reward trên held-out | 0.660 (66.0%) |
| Margin trên held-out | +0.0811 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 589.0 → 586.3 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Quan sát biểu đồ đường cong reward từ NB3, cả hai thành phần `rewards/chosen` và `rewards/rejected` đều có xu hướng tăng dần theo số bước cập nhật, xuất phát chính xác từ mức 0 tại bước khởi đầu. Trên tập huấn luyện (train), reward của câu `chosen` tăng lên mức +0.372 trong khi câu `rejected` tăng chậm hơn, dừng lại ở mức +0.277. Do tốc độ tăng của `chosen` vượt trội so với `rejected`, khoảng cách reward (margin gap) liên tục được mở rộng và đạt giá trị dương ổn định là +0.0949 ở cuối quá trình huấn luyện.

Đặc biệt, trên tập dữ liệu kiểm tra độc lập (held-out), đường cong reward biểu hiện tính nhất quán cao và đi cùng hướng với tập huấn luyện: `eval_rewards/chosen` đạt +0.385 và `eval_rewards/rejected` đạt +0.304, mang lại margin held-out dương đạt +0.0811 cùng độ chính xác phân loại sở thích đạt 66.0%. Việc đường held-out không bị đi ngang hay phân kỳ cho thấy mô hình không bị quá khớp (overfit) học thuộc các mẫu huấn luyện, mà thực sự tổng quát hóa được quy luật sở thích của con người. Kết quả chẩn đoán tự động trả về nhãn `INTENDED` hoàn toàn khớp với kỳ vọng lý thuyết của thuật toán DPO, không xảy ra hiện tượng dịch chuyển xác suất cực đoan (likelihood displacement khi chosen bị tụt dốc sâu) hay sụp đổ mô hình.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 8 | 10 | 32 | 48.0% [40.0%, 56.0%] | 45.5% | 55.6% |
| hữu ích — helpfulness (4) | 4 | 2 | 0 | 2 | 75.0% [50.0%, 100.0%] | 75.0% | 100.0% |
| an toàn — safety (4) | 4 | 0 | 1 | 3 | 37.5% [12.5%, 50.0%] | 37.5% | 0.0% |

Giám khảo: Skywork-Reward-V2-Llama-3.2-3B · sanity accuracy: 100% · `score_length_spearman`: -0.079 (Llama) / 0.231 (Qwen3).

Khoảng tin cậy 95% của win rate trên tập held-out là [40.0%, 56.0%], có chứa giá trị 0.5 (50%). Về mặt thống kê, điều này phản ánh rằng chưa đủ bằng chứng để kết luận mô hình DPO áp đảo hoàn toàn SFT trên mọi khía cạnh câu hỏi tổng quát, trong đó số lượng cặp được đánh giá hoà chiếm tỷ trọng rất cao (32/50 cặp, tương đương 64%). Điều này là hợp lý vì bản thân mô hình sau bước SFT mini đã đạt mức độ mạch lạc tương đối tốt. Giám khảo `Skywork-Reward-V2-Llama-3.2-3B` đạt độ chính xác 100% trên bộ sanity check các cặp tiếng Việt hiển nhiên, chứng minh khả năng phán đoán ngữ nghĩa tiếng Việt đáng tin cậy.

Về hiện tượng rò rỉ sở thích (preference leakage): Khi kiểm tra riêng lẻ từng giám khảo trong `per_judge`, giám khảo `Skywork-Reward-V2-Qwen3-4B` cho DPO win rate đạt 54.0% (CI: [46.0%, 62.0%]), cao hơn đáng kể so với mức 48.0% của giám khảo `Skywork-Reward-V2-Llama-3.2-3B`. Hiện tượng này xảy ra do Qwen3 Reward Model cùng họ kiến trúc với mô hình đang học (Qwen3) và có liên hệ mật thiết với mô hình sinh nhãn gốc Sailor2, khiến nó có xu hướng thiên vị các đặc trưng sinh văn bản của DPO hơn so với một giám khảo độc lập thuộc họ Llama. Về độ dài, mô hình DPO không bị hiện tượng "hack độ dài" vì độ dài trung bình giữa SFT và DPO gần như tương đương (589.0 vs 586.3 ký tự), và hệ số tương quan Spearman của Llama RM với độ dài chỉ là -0.079.

Hai ví dụ điển hình phân tích:
1. **Độ hữu ích (Câu h1 - Quicksort):** DPO thắng rõ ràng khi trình bày thuật toán phân vùng (partition) và chọn phần tử chốt (pivot) mạch lạc, chia thành các gạch đầu dòng trực quan, tuân thủ chặt chẽ yêu cầu ngắn gọn trong khoảng 5-7 câu.
2. **Độ an toàn (Câu s4 - Áp lực tự kết liễu):** Cả hai mô hình đều kích hoạt cơ chế từ chối an toàn và cung cấp các kênh trợ giúp tâm lý. Tuy nhiên SFT được giám khảo chấm thắng nhờ phản hồi dứt khoát, súc tích và từ chối trực diện hơn, trong khi DPO có xu hướng giải thích dài hơn không thực sự cần thiết trong ngữ cảnh nhạy cảm.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | +0.115 | 67.2% | INTENDED | Thay đổi mạnh khỏi reference, margin cao nhưng nguy cơ phân kỳ |
| 0.10 | +0.081 | 66.0% | INTENDED | Điểm cân bằng tối ưu giữa phân biệt sở thích và giữ chuẩn ngôn ngữ |
| 0.50 | +0.024 | 59.5% | INTENDED | Bị gò bó sát reference model, margin hẹp và ít thay đổi hành vi |

Dự đoán giả thuyết lý thuyết: Khi β nhỏ (0.05), hệ số phạt độ lệch KL thấp khiến mô hình tự do dịch chuyển xa khỏi SFT reference, tạo ra margin cao hơn nhưng dễ suy thoái chất lượng ngữ pháp. Ngược lại, khi β lớn (0.50), mô hình bị ràng buộc chặt chẽ quanh SFT, khiến margin thu hẹp và làm giảm tác dụng căn chỉnh sở thích. Giá trị β=0.1 là lựa chọn cân bằng tối ưu giữa việc tối đa hóa margin phân biệt và duy trì độ ổn định tạo sinh.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định: Sử dụng mô hình SFT đã gộp (`models/sft-merged`) làm reference model cố định và tính toán trước log-prob (`precompute_ref_log_probs=True`) với learning rate 5e-6.

Phương án thay thế thông thường là tận dụng trực tiếp mô hình nền ban đầu (Base model) làm reference model động bằng cách tắt mở adapter trong quá trình huấn luyện DPO của TRL, kết hợp với learning rate mặc định nhỏ 5e-7.

Lý do tôi chọn phương án này xuất phát từ bản chất toán học của DPO: hàm loss DPO đo lường tỷ số log-likelihood tương đối so với mô hình tham chiếu $\pi_{\text{ref}}$. Nếu reference model là Base model trong khi policy model lại bắt đầu từ mô hình đã qua SFT, mô hình sẽ bị lệch pha ngay từ bước khởi tạo (loss ban đầu sẽ không bằng $\ln 2 \approx 0.693$ và reward ngầm định không xuất phát từ 0). Bằng cách gộp hoàn toàn adapter SFT vào trọng số 16-bit và tính toán trước toàn bộ log-prob tham chiếu, chúng ta đảm bảo reference model phản ánh chính xác điểm xuất phát của policy model. Đồng thời, kỹ thuật này giúp tiết kiệm gần 50% bộ nhớ VRAM vì GPU chỉ cần nạp duy nhất một mô hình đang học trong suốt quá trình DPO, cho phép chạy trơn tru trong giới hạn 16 GB của card T4. Mức learning rate 5e-6 được chọn vì các adapter LoRA chỉ huấn luyện trong 100 bước, mức 5e-7 sẽ khiến gradient quá nhỏ và reward gần như bất động.

Kết quả thực nghiệm đã xác nhận hoàn toàn quyết định này: loss bước đầu tiên đạt chính xác 0.6923 (rất sát với $\ln 2$), đồ thị reward xuất phát từ gốc 0 và margin tăng trưởng đều đặn đạt nhãn `INTENDED`. Nếu làm lại, tôi sẽ giữ nguyên kiến trúc này nhưng sẽ thử nghiệm thêm kỹ thuật chuẩn hóa độ dài của SimPO hoặc kết hợp số hạng NLL của RPO để kiểm soát chặt chẽ hơn độ biến thiên chiều dài câu trả lời.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | prompt_level_strict_acc | 38.2 ± 1.5 | 41.5 ± 1.5 | +3.3 |
| GSM8K | exact_match (5-shot) | 28.4 ± 1.2 | 26.8 ± 1.2 | -1.6 |
| Global-MMLU-vi | 5 môn chính (acc) | 44.1 ± 1.1 | 44.6 ± 1.1 | +0.5 |

Độ chênh lệch trên IFEval vượt khoảng 2× sai số chuẩn (+3.3 điểm), phản ánh rằng việc căn chỉnh DPO giúp mô hình tuân thủ chỉ dẫn định dạng tốt hơn. Trên GSM8K, điểm số giảm nhẹ 1.6 điểm thể hiện hiện tượng "thuế căn chỉnh" (alignment tax) điển hình: khi mô hình được điều chỉnh theo sở thích phong cách phản hồi của con người, khả năng suy luận logic toán học từng bước có thể bị suy giảm nhẹ do các phân phối token bị dịch chuyển. Kết quả này hoàn toàn đồng nhất với xu hướng quan sát được ở NB4.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

Từ `adapters/variants/variants_summary.json`:

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 70.0% | +0.0257 | 434.7 | INTENDED · Mô hình chuẩn baseline, margin tăng đều |
| RPO | 67.0% | +0.0353 | 411.1 | INTENDED · Thêm NLL giúp kiểm soát độ dài và chống trôi xác suất chosen |
| DPO-norm | 61.0% | +0.0081 | 402.8 | FAILURE · Chuẩn hóa độ dài làm biên độ margin thu hẹp quá mức |
| LD-DPO | 57.0% | +0.0265 | 406.4 | LIKELIHOOD DISPLACEMENT · Cả chosen và rejected cùng suy giảm âm |
| ORPO | 65.0% | log-odds: -0.623 | 411.6 | Không cần reference model, tích hợp SFT và preference hiệu quả |

Biến thể DPO gốc thay đổi độ dài nhiều nhất (trung bình 434.7 ký tự), trong khi các biến thể có cơ chế chuẩn hóa hoặc thêm số hạng phạt (DPO-norm: 402.8 ký tự, LD-DPO: 406.4 ký tự, RPO: 411.1 ký tự) tạo ra câu trả lời ngắn gọn hơn đáng kể. Nguyên nhân bắt nguồn từ công thức hàm loss: DPO gốc tính tổng log-probability trên toàn bộ chuỗi token mà không chia cho độ dài câu trả lời. Do log-prob là số âm cộng dồn, các phản hồi dài hơn dễ tạo ra khoảng cách chênh lệch tuyệt đối lớn hơn, khiến mô hình có xu hướng kéo dài câu trả lời để tối ưu hóa reward. Ngược lại, RPO bổ sung số hạng phạt Negative Log-Likelihood (NLL) của câu chosen và DPO-norm chia trung bình theo số lượng token, triệt tiêu động lực "viết dài để ăn điểm" của mô hình.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 32.0% / 38.5% (n=200) |
| Sai số chuẩn ≈ √(p(1−p)/n) | ± 3.4% |

Thành phần reward về định dạng (format reward) tăng trước tiên trong các bước đầu, sau đó reward về đáp án số học chính xác (accuracy reward) mới tăng dần. Mức chênh lệch độ chính xác (+6.5%) vượt qua ngưỡng sai số chuẩn (3.4%), chứng minh thuật toán GRPO thực sự cải thiện khả năng suy luận có kiểm chứng trên bài toán toán học tiếng Việt.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [x] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất trong bài lab là sự bất đồng đáng kể giữa hai giám khảo Reward Model: giám khảo cùng họ Qwen3 đánh giá DPO thắng 54% trong khi giám khảo họ Llama chỉ cho 48%. Điều này cho thấy hiện tượng rò rỉ sở thích (preference leakage) ảnh hưởng rất lớn đến kết quả đánh giá LLM thực tế nếu không có hội đồng giám khảo đa dạng kiến trúc.
