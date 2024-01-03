## Python Dense Optical Flow

**Python** wrapper for Ce Liu's [C++ implementation](https://people.csail.mit.edu/celiu/OpticalFlow/) of Coarse2Fine Optical Flow. This is **super fast and accurate** optical flow method based on Coarse2Fine warping method from Thomas Brox. This python wrapper has minimal dependencies, and it also eliminates the need for C++ OpenCV library. For real time performance, one can additionally resize the images to a smaller size.

Run the following steps to download, install and demo the library:
  ```Shell
  git clone https://github.com/pathak22/pyflow.git
  cd pyflow/
  python setup.py build_ext -i
  python demo.py    # -viz option to visualize output
  ```

This wrapper code was developed as part of our [CVPR 2017 paper on Unsupervised Learning using unlabeled videos](http://cs.berkeley.edu/~pathak/unsupervised_video/). Github repository for our CVPR 17 paper is [here](https://github.com/pathak22/unsupervised-video).

-----------------------------------------------

### installation on Windows 10
 1. Install Microsoft Visual C++ Build Tools 설치
   - Visual Studio Community를 설치하면서, Desktop용 C++ Build package를 추가로 설치 (~6 GB)
   - 관리자 권한으로 실행 된 CMD를 통해 `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build`로 이동 후 운영체제에 맞는 `vcvars*.bat` 파일을 실행
    e.g.) 64bit - vavars64.bat
   - 아래와 같은 에러가 발생할 수 있음
    b) `cl.exe` 를 실행할 수 없음 외 Visual Studio 를 찾을 수 없는 경우
     sol)
      `vcvars*.bat` 파일이 실행되지 않은 경우.
    a) `[ERROR:VsDevCmd.bat] Script "vsdevcmd\ext\잘못된" could not be found.`
     sol) 
      `VsDevCmd.bat` 파일 내에서 loop를 돌릴 때 `vsdevcmd\ext` 폴더 내의 bat 파일만 돌아야 하지만 어떤 이유에서인지 첫 파일로 `잘못된`을 인식하는 에러가 발생하는 것을 확인함.
      첫 번째 루프는 스킵하는 방식으로 버그를 우회함.
      `C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools` 폴더 내의 `VsDevCmd.bat` 파일을 바탕화면으로 복사 후 메모장으로 편집 (백업 필요)
      `128`번 줄을 다음과 같이 수정
      ```bat
         for /F %%a in ( 'dir "%VS170COMNTOOLS%vsdevcmd\ext\*.bat" /b /a-d-h /on' ) do (
         
         for /F "skip=1 delims=" %%a in ( 'dir "%VS170COMNTOOLS%vsdevcmd\ext\*.bat" /b /a-d-h /on' ) do (
      ```
      테스트용 bat file을 만들어 위 루프를 실행하며, 변수 `%%a`가 어떤 값을 반환하는지 확인할 필요가 있음. 외국 커뮤니티의 경우 invalid, active 등 다양한 변수 값으로 인해 버그가 발생하는 사례를 확인함.
    
 2. Pyflow 설치
   - 기본적으로 Linux용으로 개발된 버전이라 수정이 필요함.
   - `src\project.h` 파일 내에서 9번 줄의 `#define _LINUX_MAC` 를 comment 처리.
   - 위에 지시된 대로 `setup.py`를 실행시켜 설치.
   - 아래와 같은 에러가 발생할 수 있음
    a) `error C2146: 구문 오류: ')'이(가) 'a' 식별자 앞에 없습니다.`
       `'T1': 재정의: 이전 정의는 '템플릿 매개 변수'입니다.`
       `...`
     sol)
      위에서 말한 `#define _LINUX_MAC`을 comment 처리 하지 않을 때 발생.
      pyflow github issue 를 보고 해결함
    b) `pyflow.obj : error LNK2001: 확인할 수 없는 외부 기호 __imp__PyBaseObject_Type`
       `pyflow.obj : error LNK2001: 확인할 수 없는 외부 기호 __imp___PyDict_NewPresized`
       `...`
     sol)
      OS에 맞지 않는 `vcvars.bat` 파일을 실행했을 때 발생.
      다른 vcvars.bat 파일을 실행하여 재시도.
    c) `LINK : fatal error LNK1158: 'rc.exe'을(를) 실행할 수 없습니다.`
     sol)
      `rc.exe` 파일을 제대로 인식하지 못해 발생하는 문제.
      Visual Stduio가 제대로 설치됐다면 Windows SDK / Windows Kit이 설치됐을테지만, Path에 추가되지 않아 발생하는 문제.
      해당 위치에서 `rc.exe` 파일을 복사하여 해결할 수 있음
      `C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x64` (64bit OS, SDK ver 10.0.19) 에서 `rc.exe` 및 `rcdll.dll` 복사
      `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.33.31629\bin\Hostx64\x64` (cl.exe가 있는 위치, 64bit OS, `vcvars64.bat` 실행)에 복제 
