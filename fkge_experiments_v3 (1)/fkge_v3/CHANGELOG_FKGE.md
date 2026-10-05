# FKG-E v3 — cài đặt lại theo Chương 3

Bản v2 (`models/fkge_old_v2.py`, giữ để đối chiếu) khác Chương 3 ở nhiều điểm: skip-gram theo cửa sổ vị trí,
L_node so khớp trọng số μ của tệp cạnh, p_E là trộn phân phối one-hot của luật, không có L_edge/L_A/L_B,
L_rule là SGNS, không có λ_C, không chiếu, Adam/SGD tự viết, trọng số đặt tay (δ_pred = 20).

`models/fkge.py` (mới) cài đặt đúng các định nghĩa và thuật toán của Chương 3 (xem README). Unit test
`tests/test_fkge_ch3.py` (13 test, đều đạt) kiểm tra:
1. ma trận đồng xuất hiện toàn luật, Mệnh đề đồng xuất hiện ↔ (ā, b);
2. P_n, ω đối xứng và thuộc (0, 1];
3. nhúng luật (3.pool), nhúng truy vấn (3.hj)–(3.query), gộp đều α = r/(r+1);
4. p_E là phân phối, ≥ ε/n_L; cận của log-sum-exp; cos_ε0 ∈ (−1, 1);
5. miền giá trị các thành phần, cận dưới L_B, cận L_inf, L_rule, cận c_τ của L_pred;
6. cận dưới L_B đạt được với b̃ (Mệnh đề 3.coer(iii));
7. bất biến quay L(QΘ) = L(Θ);
8. tính bức L ≥ λ_C‖Θ‖², phép chiếu giữ ‖Θ‖ ≤ B₀, λ_C = 0 bị từ chối;
9. ước lượng theo lô không chệch (Thuật toán 3.10);
10. gradient trùng sai phân hữu hạn;
11. giáo viên: argmax p_F^cal trùng quyết định của FISA, e^F chỉ trên luật của ŷ_F;
12. huấn luyện giảm hàm mục tiêu và chuẩn gradient; a_E là phân phối trên luật của nhãn dự đoán;
    Dev ≥ tỉ lệ bất đồng (Mệnh đề quan hệ giữa các ràng buộc);
13. ánh xạ tên tham số cũ.
