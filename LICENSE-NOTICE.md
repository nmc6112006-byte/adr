# License Notice — Dữ liệu corpus mẫu

## Corpus YouMed (thư mục `assets/sample_corpus/`)

- **Nguồn chính thức (paper):** https://huggingface.co/datasets/vuduylinh150804/vietmed-crawled-corpus
- **Bản gốc:** các bài viết y tế từ nền tảng YouMed (https://youmed.vn), URL nguồn
  được ghi ở dòng đầu mỗi file (`# SOURCE_URL:`).
- **License:** CC-BY-NC-4.0 (theo khai báo của dataset trên HuggingFace).
- **Điều kiện sử dụng:** ghi công tác giả/nền tảng YouMed + dataset ViMedGraph,
  KHÔNG sử dụng cho mục đích thương mại.

## Trích dẫn paper nguồn

> Ha Nguyen-Tien, Linh Vu-Duy, Minh-Huyen Nguyen-Thi, Toan Le-Ngoc, Dinh-Thai Kim.
> "Adaptive Directed Graph Retrieval for Graph-Based Retrieval-Augmented Generation."
> Applied Intelligence (2025/2026). Code chính thức:
> https://github.com/Jmhzbmcn2/Graduation-Thesis · Supplementary:
> https://github.com/Jmhzbmcn2/adgr-supplementary

## Bộ QA đánh giá (`assets/qa_eval_sample.json`)

16 câu hỏi kèm đáp án tham chiếu được biên soạn dựa trên nội dung các bài
YouMed trong sample corpus, mô phỏng phương pháp gán gold-chunk của paper
(Section 5.3). Mỗi item ghi rõ `gold_file` + `gold_section` để ánh xạ gold chunk.
