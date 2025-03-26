# Design and Implementation of a Text Dependent Speaker Verification for

## Abstract

화자 검증(Speaker Verification, SV)는 입력된 음성 발화와 사전에 등록된 화자의 음성 특징을 비교하여 신원을 검증하는 기술로, 텍스트 독립 방식(Text Independent SV, TI-SV)과 텍스트 종속 방식(Text Dependent SV, TD-SV)으로 구분된다. 

TI-SV는 발화 내용에 제약이 없다는 장점이 있으나, 다양한 음운적 변이로 인해 정확한 식별이 어렵고, 짧은 발화 환경에서는 성능 저하가 발생하는 한계를 가진다.
이에 반해 TD-SV는 제한된 음소 구조를 기반으로 하여 짧은 발화에서도 높은 성능을 보이며, 발화 내용과 화자 정보를 동시에 활용할 수 있어 인증의 신뢰성을 높일 수 있다.

그러나 실제 응용 환경에서는 등록(Enrollment)과 테스트(Test) 과정 간의 녹음 거리, 채널, 잡음 등 도메인 불일치 문제가 발생하기 쉽고, 이는 전체 시스템 성능 저하로 이어질 수 있다.

본 연구에서는 이러한 환경적 불일치에 강건하면서도 짧은 발화 기반으로 고신뢰 음성 인증이 가능한 TD-SV 시스템을 설계하고, 이를 위해 Conformer 기반 딥러닝 모델 및 다양한 최적화 기법을 통합하여 제안한다.

### keyword
Speaker verification, text-dependent speaker verification, end-to-end architecture, conformer, domain mismatch, noise robustness, transfer learning
