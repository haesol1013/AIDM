import numpy as np

scores = np.load("scores.npy")

q1 = np.mean(scores, axis=(1, 2, 3))
print("Q1) 학년 별 전 과목 평균 점수: ")
for i, v in enumerate(q1):
    print(f"{i+1}학년 평균 점수: {v}")

q2 = np.mean(scores, axis=(0, 1, 2))
subjects = ["국어", "영어", "수학", "사회", "과학"]
print("\nQ2) 과목 별 전체 평균 점수: ")
for subject, v in zip(subjects, q2):
    print(f"{subject} 평균 점수: {v}")

q3 = np.mean(scores[2, :, :, 2], axis=(0, 1))
print("\nQ3) 3학년 학생들의 평균 수학 점수:", q3)

q4_target = np.mean(scores[3:], axis=3)
q4_condition = q4_target >= 90
q4_answer = q4_condition.sum()
print("\nQ4) 4, 5, 6학년 학생 중 평균 점수가 90점 이상인 학생 수:", q4_answer)


