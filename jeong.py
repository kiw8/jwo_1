import numpy as np

# Softmax 활성화 함수 정의
def softmax(logits):
    exp_values = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Overflow 방지
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities

# 교차 엔트로피 손실 함수 클래스 정의
class CrossEntropy:
    def forward(self, predictions, targets):

        # 예측 값을 소프트맥스 활성화 함수로 변환
        predictions = softmax(predictions)

        # 예측 값이 0이 되는 것을 방지하기 위해 작은 값으로 클리핑
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)

        # 타겟이 원-핫 인코딩이 아닐 경우
        if targets.ndim == 1:
            correct_confidences = predictions[np.arange(len(predictions)), targets]
        else:
            # 원-핫 인코딩 처리
            correct_confidences = np.sum(predictions * targets, axis=1)

        # 음의 로그 우도 계산
        negative_log_likelihoods = -np.log(correct_confidences)

        return np.mean(negative_log_likelihoods)

# 예시 사용
logits = np.array([
    [2.0, 1.0, 0.1],  # 첫 번째 데이터의 로짓 값
    [1.0, 3.0, 0.2],  # 두 번째 데이터의 로짓 값
    [0.5, 0.7, 2.5]   # 세 번째 데이터의 로짓 값
])

# 실제 정답 레이블 (0, 1, 2)
targets = np.array([0, 1, 2])

# 교차 엔트로피 손실 클래스 생성
ce = CrossEntropy()

# 손실 계산
loss = ce.forward(logits, targets)
print("Categorical Cross-Entropy Loss:", loss