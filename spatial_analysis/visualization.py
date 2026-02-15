# -*- coding: utf-8 -*-
"""
Visualization Module

시각화 관련 유틸리티 함수들을 제공합니다.
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings


def setup_korean_font():
    """시스템에 설치된 한글 폰트를 찾아서 설정"""
    # 가능한 한글 폰트 후보들
    font_candidates = [
        "NanumBarunGothic", 
        "NanumGothic", 
        "NanumBarunGothicOTF",
        "Noto Sans CJK KR", 
        "Noto Sans CJK", 
        "Noto Sans",
        "Malgun Gothic",  # Windows
        "AppleGothic",    # macOS
        "NanumMyeongjo",
        "NanumSquare",
        "NanumPen",
        "DejaVu Sans"     # fallback
    ]
    
    # 시스템 폰트 경로
    font_path = None
    font_name = None
    
    # 시스템 폰트 목록 가져오기
    system_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 후보 폰트 중 시스템에 있는 것 찾기
    for candidate in font_candidates:
        if candidate in system_fonts:
            font_name = candidate
            # 실제 폰트 파일 경로 찾기
            for font_file in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
                if candidate in font_file:
                    font_path = font_file
                    break
            if font_path:
                break
    
    if font_name and font_path:
        try:
            # 폰트 설정
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지
            print(f"Font set: {font_name}")
            print(f"  path: {font_path}")
        except Exception as e:
            print(f"Font setup error: {e}")
            print(f"  Using default font.")
            plt.rcParams['axes.unicode_minus'] = False
    else:
        # 폰트를 찾지 못한 경우, matplotlib의 기본 폰트를 사용하되 한글 경고 무시
        print("Korean font not found. Using default font.")
        print("  To fix: sudo apt-get install fonts-nanum")
        plt.rcParams['axes.unicode_minus'] = False
        # 한글 깨짐 경고 무시 설정
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    return font_name
