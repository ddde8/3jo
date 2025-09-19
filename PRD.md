# 프로젝트 PRD (Product Requirements Document)

## 1. 프로젝트 개요
- **목표**: 버튼을 통해 영상 화면(videoSprite)과 통계 화면(graphSprite)을 전환할 수 있는 OpenCV 기반 인터랙티브 애플리케이션 개발

## 2. 주요 요구사항
- 버튼 클릭 시 영상/통계 화면 전환
- 영상 화면: 실시간 웹캠 영상 (videoSprite 활용)
- 통계 화면: 예시 그래프(임의 데이터, graphSprite 활용)
- 직관적인 UI, 마우스 클릭으로 화면 전환

## 3. 주요 기능
- videoSprite: 웹캠 영상 표시
- graphSprite: matplotlib 등으로 생성한 그래프 이미지 표시
- ButtonSprite: 화면 전환 버튼

## 4. 화면 설계
- 상단: 화면 전환 버튼, 시간 표시
- 중앙: videoSprite 또는 graphSprite 중 하나만 표시

## 5. 기술 스택
- Python, OpenCV, numpy, matplotlib

## 6. 기타
- 확장 가능 구조(추후 다양한 화면 추가 가능)
