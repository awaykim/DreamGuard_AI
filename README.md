# ECG를 통한 OSA 진단 (위험점수 및 상위 백분위 계측) AI 모델 서빙 API

### 방법
1. 가상환경에 requirements.txt 내 라이브러리 설치
2. 가상환경 실행
3. 작업 디렉토리를 ECG_AI(app.py가 있는 폴더)로 이동
4. 터미널에서 python app.py
   - 엔드포인트(로컬): http://localhost:5000/upload 
   - 파일 업로드(ZIP 형식) - **data 폴더의 a01.zip**
   - 업로드하면 API에서 압축 해제 uploads 폴더에 파일 저장
   - upload 폴더의 파일을 통해 AI  추론
   - 위험점수(Prob)과 상위 백분위(Percentile)를 JSON 형태로 반환함
   -  postman에서 작동 확인 **POST만 됨**
   - ![image](https://github.com/jeaniejan/Capston/assets/121528605/1a75357f-d788-49cf-a668-16d500fceee0)
   - 웹(크롬 등)에서 접속 시 오류 나는거 맞음!
### 기타
- 속도가 느림: 6분에서 7분 대
     - 아마 로컬 CPU를 쓰다보니 추론이 아주 오래걸리는 것 같음
     - 속도 최적화를 하는 방안을 찾아보겠음!
