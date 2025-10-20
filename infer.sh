#!/usr/bin/env bash

# python -m src.inferences \
#   --adapter ./output/nemotron-sft-trained/epoch-2 \
#   --question "Bạn là một trợ lý AI y khoa đáng tin cậy được phát triển để hỗ trợ phân tích và đưa ra lời khuyên y tế chính xác.\n\nHãy phân tích câu hỏi sau một cách logic, khách quan và theo văn phong chuyên ngành y. Chỉ dựa vào thông tin có trong câu hỏi để suy luận, không tự tạo giả định hoặc phỏng đoán mơ hồ.\n Nếu câu hỏi liên quan đến chính trị, thông tin sai lệch, nội dung tiêu cực, hoặc đi ngược lại đạo đức y khoa – hãy từ chối trả lời để đảm bảo an toàn, bảo mật và tuân thủ quy tắc chuyên môn.\n\n### Câu hỏi :\nMột người đã hoàn thành việc tiêm phòng uốn ván cách đây 10 năm. Nếu họ hiện có một vết thương sạch được tạo ra cách đây 2,5 giờ, họ nên nhận được điều trị y tế nào? " \
#   --max-length 1024 \
#   --temperature 0.3 \
#   --top-p 0.9 \
#   --load-in-4bit

# python -m src.inferences \
#   --adapter ./output/nemotron-sft-trained/epoch-2 \
#   --question "Thủ đô của Việt Nam là gì?" \
#   --max-length 2048 \
#   --temperature 0.3 \
#   --top-p 0.9 \
#   --load-in-4bit

python -m src.inferences \
  --adapter ./output/nemotron-sft-test_mini_en/epoch-19 \
  --question "A person completed tetanus vaccination 10 years ago and now has a clean wound created 2.5 hours agoy " \
  --max-length 256 \
  --device cuda


  # --temperature 0.6 \
  # --top-p 0.9 \
  # --load-in-4bit \

# python -m src.inferences \
#   --adapter ./output/nemotron-sft-test_mini_en/epoch-17 \
#   --question "Patient with macroglossia, atrophic papillae, Hgb 11.5 g/dL, MCV 100 fL. Next best step?" \
#   --max-length 1024 \
#   --temperature 0.3 \
#   --top-p 0.9 \
#   --repetition-penalty 1.1 \
#   --load-in-4bit