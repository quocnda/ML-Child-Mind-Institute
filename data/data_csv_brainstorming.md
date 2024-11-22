1. Link data đầy đủ: [Data of child mind institute](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data)
2. Bối cảnh là tổ chức 1 cuộc khảo sát, flow sẽ gồm 2 pha:
    - Enrollment (Ghi danh): participant sẽ ghi danh vào việc sẽ tham gia cuộc khảo sát.
    - Participation (Tham gia): tham gia vào thực hiện khảo sát thực tế. Trong cuộc khảo sát này, các bài test sẽ được đưa ra vào thời gian khác nhau (trong data thì thời gian ứng với mỗi bài test là season).
3. Phân tích một số file csv quan trọng
# Data dictionary
- Là từ điển dữ liệu, cung cấp **thông tin về các field** được dùng.
## Các cột:
    1. Instrument: danh mục mà field thuộc về
    2. Field
    3. Description: ý nghĩa của field
    4. Type: Data type của field như float, str(ing), categorical int
    5. Values: Các giá trị có thể (Thường áp dụng cho type là string/ categorical int)
    6. Value Labels: Ý nghĩa của values (VD: 0 -> chết; 1 -> sống)
## Các dòng:
- Phân tích chia theo nhóm các field (instrument)
- Định dạng gọn theo thứ tự cột *<field, des, type, (values), (value labels)>*
### 1. Identifier
- <id, định danh participant, str>
### 2. Demographics:
- <Basic_Demos-Enroll_Season, mùa ghi danh, str, (xuân, hạ, thu, đông)>
- <Basic_Demos-Age, tuổi participant, float>
- <Basic_Demos-Sex, giới tính participant, categorical int, (0, 1), (0: trai, 1: gái)>
### 3. Children's Global Assessment Scale: 
Đo sức khỏe tình thần thông qua việc đo chức năng tâm lý xã hội
- CGAS-Season: mùa mà participant thực hiện bài test
- CGAS-CGAS_Score
### 4. Physical Measure
- Physical-Season
- Physical-BMI
- Physical-Height
- Physical-Weight
- Physical-Waist_Circumfrence
- Physical-Diastolic_BP, huyết áp tâm thu (áp lực máu trong động mạch khi tim bóp)
- Physical-HeartRate
- Physical-Systolic_BP, huyết áp tâm trương ("" tim nghỉ khi giữa 2 lần bóp)
### 5. FitnessGram Vitals and Treadmill: 
Test sức bền thông qua máy chạy bộ.
- Fitness_Endurance-Season: mùa mà participant thực hiện bài test
- Fitness_Endurance-Max_Stage: độ khó cao nhất của bài test (chẳng hạn như độ nghiêng của treadmill) mà participant đạt được
- Fitness_Endurance-Time_Min: Thời gian participant chạy duy trì.
- Fitness_Endurance-Time_sec: time dư theo giây?
### 6. FitnessGram Child: 
FGC-FGC_x là sample còn FGC-FGC_x_Zone là space
- FGC-Season
- FGC-FGC_CU: curl up (gập bụng), int
- FGC-FGC_CU_Zone, categorical int, (0: needs improvement, 1: healthy)
- FGC-FGC_GSND: grip strength non dominant (bóp năng cầm lực bằng tay ko thuận), float
- FGC-FGC_GSND_Zone, categorical int, (1: weak, 2: normal, 3: strong)
- FGC-FGC_GSD: grip strength dominant (bóp năng cầm lực bằng tay thuận)
- FGC-FGC_GSD_Zone
- FGC-FGC_PU: push up 
- FGC-FGC_PU_Zone
- FGC-FGC_SRL: Sit & reach left (ngồi thẳng 2 chân, gập người)
- FGC-FGC_SRL_Zone
- FGC-FGC_SRR: Sit & reach right
- FGC-FGC_SRR_Zone
- FGC-FGC_TL: trunk lift (nằm sấp, vươn cổ & lưng dậy)
- FGC-FGC_TL_Zone
### 7. Bio-electric Impedance Analysis: 
Phép đo sử dụng dòng điện để đo chỉ số cơ thể như tỷ lệ chất béo, nước,khối lượng cơ 
- BIA-Season
- BIA-BIA_Activity_Level_Num, mức độ hoạt động thể chất, categorical int, (1-5), (1=Very Light, 2=Light, 3=Moderate, 4=Heavy, 5=Exceptional) 
- BIA-BIA_BMC: bone mineral content: khoáng chất trong xương -> đánh giá sức khỏe của xương
- BIA-BIA_BMI: body mass index -> mức độ thừa/ thiếu cân
- BIA-BIA_BMR: basal metabolic rate: mức độ chuyển hóa cơ bản -> năng lượng mà cơ thể tiêu hao để duy trì sự sống cơ bản
- BIA-BIA_DEE: daily energy expenditure: tổng calo cơ thể tiêu mỗi ngày
- BIA-BIA_ECW: extracellular water: lượng nước ngoài tế bào (huyết tương, v.v.) -> tình trạng cân bằng nước của cơ thể
- BIA-BIA_FFM: fat-free mass: khối lượng cơ thể trừ mỡ -> đo sức khỏe cơ bắp
- BIA-BIA_FFMI: fat-free mass index: tỷ lệ FFM với chiều cao
- BIA-BIA_Fat: lượng mỡ cơ thể
- BIA-BIA_Frame_num: khung xương cơ thể, categorical int, (1=small, 2=medium, 3=large)
- BIA-BIA_ICW: intracellular water: lượng nước trong tế bào
- BIA-BIA_LDM: lean dry mass: 1 phần của FFM
- BIA-BIA_LST: lean soft tissue: mô mềm (cơ, v.v.) ko mỡ -> đánh giá lượng cơ bắp
- BIA-BIA_SMM: skeletal muscle mass: tổng lượng cơ xương
- BIA-BIA_TBW: total body water: tổng ICW (trong tế bào) và ECW (ngoài tế bào)
### 8. Physical Activity Questionaire (Adolescents)
- PAQ_A-Season
- PAQ_A-PAQ_A_Total, mức độ hoạt động thể chất, float
### 9. Physical Activity Questionaire (Children)
- PAQ_C-Season
- PAQ_C-PAQ_A_Total
### 10. Parent-Child Internet Addiction Test
20 bài test dành cho parent để trả lời về con cái của họ, đáp án theo type là categorical int, tức là đưa ra các mức độ 0-5 (ý chỉ tần suất). Trong đó: 0=Does Not Apply, 1=Rarely, 2=Occasionally, 3=Frequently, 4=Often, 5=Always
- PCIAT-Season
- PCIAT-PCIAT_01: ko tuân theo online use time limits ko?
- PCIAT-PCIAT_02: lơ việc nhà để dành thêm cho online time ko?
- PCIAT-PCIAT_03: thích online hơn là chơi vs gia đình ko?
- PCIAT-PCIAT_04: tạo mqh online ko?
- PCIAT-PCIAT_05: parent phàn nàn child về vấn đề lượng time dành cho online ko?
- PCIAT-PCIAT_06: điểm thấp vì online ko?
- PCIAT-PCIAT_07: check email trước khi làm gì khác ko?
- PCIAT-PCIAT_08: từ bỏ những việc khác (mqh khác?) từ khi phát hiện ra internet ko?
- PCIAT-PCIAT_09: trở nên dè dặt, chống đối khi hỏi về việc làm gì trên internet ko?
- PCIAT-PCIAT_10: bắt gặp lén dùng internet ko như theo ý parent muốn ko?
- PCIAT-PCIAT_11: chơi máy tính 1 mình trong phòng ko?
- PCIAT-PCIAT_12: nhận call lạ từ online friends ko?
- PCIAT-PCIAT_13: hành xử mất dạy khi bị làm phiền khi đang online ko?
- PCIAT-PCIAT_14: cảm thấy mệt mỏi, kiệt sức hơn kể từ khi bắt đầu sử dụng internet ko?
- PCIAT-PCIAT_15: cảm thấy lo lắng khi ko được dùng internet và chỉ muốn quay lại khi offline ko?
- PCIAT-PCIAT_16: nổi giận khi parent can thiệp vào thời gian sử dụng internet ko?
- PCIAT-PCIAT_17: chọn dành thời gian online thay vì tham gia vào các sở thích hoặc hoạt động ngoài trời trước đây k0?
- PCIAT-PCIAT_18: trở nên giận dữ hoặc khó chịu khi parent đặt giới hạn về thời gian sử dụng internet k0?
- PCIAT-PCIAT_19: dành nhiều thời gian online hơn là đi chơi với bạn bè không?
- PCIAT-PCIAT_20: cảm thấy buồn bã, lo lắng hoặc bực bội khi offline và cảm thấy dễ chịu khi quay lại online ko?
### 11. Sleep Disturbance Scale
- SDS-Season
- SDS-SDS_Total_Raw
- SDS-SDS_Total_T: T-score
### 12. Internet Use
- PreInt_EduHx-Season
- PreInt_EduHx-computerinternet_hoursday, số giờ dùng internet/computer, categorical int, (0, 1, 2, 3), (0: < 1h/ngày, 1: ~ 1h/ngày, 2: ~2h/ngày 3: > 3h/ngày)
# Train

# Sample submission