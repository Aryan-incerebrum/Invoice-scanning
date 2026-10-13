# Eval report

- Invoices: 6
- OCR engine: rapidocr
- LLM: qwen2.5:7b
- Review rate: 1.0
- Silent error rate (R6, headline metric): 0.0
- Line-item row match rate: None
- Median seconds/invoice: 19.1  (p95: 22.3)

## Header field accuracy

- vendor_name: 1.0
- invoice_number: 1.0
- invoice_date: 1.0
- currency: 1.0
- total: 1.0

## Per-invoice

- img1.jpeg: needs_review, header_correct=True, 16.6s
- img2.jpeg: needs_review, header_correct=True, 15.1s
- img3.jpeg: needs_review, header_correct=True, 22.0s
- img4.jpeg: needs_review, header_correct=True, 19.0s
- img5.jpeg: needs_review, header_correct=True, 19.2s
- img6.jpeg: needs_review, header_correct=True, 22.3s