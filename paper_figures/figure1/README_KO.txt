Figure 1 재제작 (NMI 재투고용)
2026년 10월 8일

산출물
Figure1_NMI_v2.pptx     편집용 PPT. 슬라이드 14.4 × 12.5 in(2배 스케일), 인쇄 183 × 159 mm.
                        글자 전부 텍스트 박스, 선·화살표·박스·히스토그램 막대·브래킷 전부 도형, 이미지 10장.
                        2차 수정(10/7)에서 글자를 라벨 수준으로 줄였고, 3차(10/8)에서 데이터 도식 세 개, 4차(10/8)에서 공통 단계 헤더·열 정렬,
                        화학 오차 막대, 지도학습 정책 분포(새 생성)를 추가함. 5차(10/8)에서 3D 렌더를 Blender Cycles로 교체하고,
                        OpenMC 출력을 열중성자속·핀 핵분열률 2D 정사각형 두 장으로 바꾸고, 데이터 차트를 matplotlib 이미지로 구움.
                        조건·수치 세부는 fig1_caption.tex로 옮김.
Figure1_NMI_v2.pdf      LibreOffice로 변환한 PDF. Arial Regular/Bold/Italic TrueType 임베드, Type 3 없음.
Figure1_NMI_v2_preview.png   150 dpi 미리보기.
fig1_caption.tex        캡션 초안. 라벨은 기존 fig:concept 그대로.

타이포그래피
본문 글자 Arial 7 pt(파일 안에서는 14 pt). 눈금 숫자·컬러바 수치·보조 문구(lower is better, dashed … uniform, k_eff)는 6 pt. 패널 문자 a b c만 8 pt 볼드.
데이터 차트(chart_*.png)는 make_charts.py가 같은 Arial 7/6 pt로 인쇄 크기 그대로 그려서 600 dpi PNG로 구운 것이라, 행 라벨과 점·막대의 정렬이 뷰어에 관계없이 고정됨.
차트 안 글자를 바꾸려면 make_charts.py를 고치고 다시 실행. 패널 제목과 attribution 세 단어(Availability, Redistribution, Information value), 박스 첫 줄, 인셋 제목만 볼드. 글자색은 한 가지.

색 사용
OpenMC 열중성자속(magma)·핀 핵분열률(viridis), 측정 수율 히트맵(cividis), 그리고 3D 렌더와 같은 Gd 봉 재질색(0072B2)뿐. 나머지는 검정·회색·흰색.

구성
맨 위 헤더 한 줄이 두 구현의 공통 단계(training corpus → policy → candidate → scientific evaluator)를 적고, a와 b는 같은 열 격자에 놓임.
a  훈련 격자 3D → 17×17 단면 → 289 심볼 → Gemma 3 정책 → 생성 텍스트 → 생성 격자 3D(DPO 선택 설계, 33개) → OpenMC 장 3D → 목적함수 → DPO/GRPO 업데이트
   왼쪽 아래에 코퍼스 k_eff 히스토그램(reactor_10k_final.csv, 1.162–1.215)과 타깃 1.05, 간격 11,192 pcm.
b  훈련 코퍼스 격자(15 task × 4 ligand, anchor = t mod 4, 정렬된 식별자 기준) → categorical 정책 → Buchwald–Hartwig 반응식(RDKit) → 측정 수율 히트맵 → 보상 → 업데이트
c  inventory 축 위의 attribution. 지도학습 정책 5,000개 출력의 윤곽 히스토그램(16에서 클리핑, 88.6%가 정확히 16), 코퍼스는 축 위 삼각형,
   정렬 정책 10,000개 출력 히스토그램(t6 감사), uniform 샘플러 밀도선(1/265, 같은 스케일),
   28–35 브래킷(선택 설계 inventory), informed 20–40 브래킷, 목적항 제거 화살표. 프레임워크 세 단어는 해당 요소 옆에 직접 붙임
   (Availability=uniform 밀도선, Redistribution=정렬 정책 히스토그램, Information value=informed 브래킷).
   오른쪽 위는 best objective(2,000회, 5런 평균±SD: uniform 1.767, informed 1.653, DPO 1.654, GRPO 1.655),
   오른쪽 아래는 화학 off-corpus 확률(0.29/0.70/0.71, uniform 0.75)과 expected yield error(50.0/29.9/29.8, uniform 57.1) 막대.

자산 출처 (assets/, data/fig1_assets.json)
blender_input/output.png           render_blender.py(Blender 5.2 Cycles, Metal GPU, 256 samples, 2000×3400, 투명 배경)로 새로 렌더.
                                   기하는 revised_nature_3/render_assembly2.py와 동일(피치 단위, 34피치 단축). input=pattern 1029(16 Gd), output=DPO best(33 Gd) 중간 절단.
assembly_input/output/fields.png   이전 PyVista 렌더(예비용). blender_*.png가 없으면 make_fig1.js가 이걸 씀.
chart_fields.png                   OpenMC 열중성자속(170×170, E<0.625 eV, σ=0.8 평활, 최댓값 기준)과 핀 핵분열률(289 cell 평균 기준, 최대 1.55) 정사각형 두 장.
                                   fields_dpo_best_hi.npz(k=1.0492, 100만 입자). 수치는 data/fields_chart.json.
lattice_input.png                  claude_figure/revised_fable1/fig1_data.json의 예시 격자(pattern 1029, 16 absorber).
mol_toluidine.png, mol_product.png RDKit 2026.03, Arial, 600 dpi 기준 1:1 픽셀로 렌더(글자 7 pt, 결합선 0.6 pt).
yield_heatmap.png                  empirical/data/buchwald_hartwig.csv. 결측 없는 4×22 슬라이스 중 수율 범위가 가장 넓은
                                   (aryl halide, base) 조합 = 3-bromopyridine + MTBD. 행은 데이터셋 순서 XPhos, t-BuXPhos,
                                   t-BuBrettPhos, AdBrettPhos, 열은 additive 데이터셋 순서. 0–100 % 고정 스케일.
히스토그램                           professor/Test/t6_generation_yield/t6_samples_v2.csv의 gd_count 10,000개(DPO+GRPO).
화학 확률                            NMI_submit/1st appeal/figure/fig4_chemistry_plotted_values.csv의 expansion_mass
                                   (sft_base 0.287, dpo 0.701, grpo 0.714, random_full 0.750).
컬러바 수치                          revised_nature_3/fields_stats.json(peak_rel 1.417).
지도학습 정책 분포                   gen_supervised_policy.py로 새로 생성(10/8). after/training/base/sft/final_sft_model_seed{0..4}에서
                                   1.05 프롬프트, temperature 1.0, 400-token cap, EOS(+generation_config의 top-k 64, top-p 0.95), 1,000개씩.
                                   파싱·보정은 t6 감사와 동일. 결과 data/supervised_generation.csv, 요약 data/supervised_generation_summary.json.
                                   seed별 정확히 16 비율 99.5 / 99.8 / 99.7 / 44.8 / 99.1 %(seed 3은 15가 많음), 전체 88.6 %, 20개 이상 0건. OpenMC 미실행.

재생성
0) 3D 렌더:  /Applications/Blender.app/Contents/MacOS/Blender -b --python render_blender.py -- --mode input --out assets/blender_input_raw.png
   (output 모드도 같은 방식, 이후 PIL로 투명 여백 trim → assets/blender_*.png. 모드당 약 1분)
   차트:  python3 make_charts.py   (matplotlib, scipy, pandas)
1) 자산:  python make_assets.py   (rdkit, matplotlib, numpy, pandas, pillow, tqdm 필요. venv 권장)
   추가 수치:  python3 make_assets_extra.py   (코퍼스 k 히스토그램, anchor 격자, best objective. pandas만 필요)
   지도학습 정책 생성:  envs/openmc의 python gen_supervised_policy.py   (torch+transformers, MPS, 약 30분)
2) 슬라이드:  node make_fig1.js   (pptxgenjs 필요. 설치 위치는 환경변수 PPTXGENJS로 지정,
   예: PPTXGENJS=/path/to/node_modules/pptxgenjs node make_fig1.js)
3) PDF:  soffice --headless --convert-to pdf Figure1_NMI_v2.pptx
4) 미리보기:  pdftoppm -r 150 -png -singlefile Figure1_NMI_v2.pdf Figure1_NMI_v2_preview

검수
600 dpi로 렌더해 패널별 크롭으로 겹침·관통·줄바꿈을 확인함. 마지막 빌드에서 결함 없음.
PPT를 PowerPoint에서 열면 LibreOffice 렌더와 줄바꿈이 미세하게 다를 수 있음. 텍스트 박스 폭은 여유를 두었음.

기존 파일은 건드리지 않음. nmi_submission/의 Figure1_NMI.* 와 fig1_concept.png는 그대로임.
