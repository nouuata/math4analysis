import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import os
from glob import glob
import cv2

plot_fld = 'plots/'

if not os.path.exists(plot_fld):
    os.makedirs(plot_fld)
plt_art_fld = plot_fld + 'article/'
if not os.path.exists(plt_art_fld):
    os.makedirs(plt_art_fld)

cdf_fld = plot_fld + 'cdf/'
jnt_fld = plot_fld + 'joint_dist/'

A = cv2.imread(cdf_fld+'cdf_Резултат_Олимпиада_Брой_участници.jpg')
B = cv2.imread(cdf_fld+'cdf_Резултат_ПМС_Брой_участници.jpg')
C = cv2.imread(cdf_fld+'cdf_Резултат_Математика_за_Всеки_Брой_участници.jpg')
D = cv2.imread(cdf_fld+'cdf_Резултат_Кенгуру_Брой_участници.jpg')
row1 = cv2.hconcat([A,B])
row2 = cv2.hconcat([C,D])
img = cv2.vconcat([row1,row2])
cv2.imwrite(plt_art_fld + 'cdf_distributions.jpg', img)

A = cv2.imread(jnt_fld+'aperc_Олимпиада_ПМС.jpg')
B = cv2.imread(jnt_fld+'aperc_Олимпиада_Математика_за_Всеки.jpg')
C = cv2.imread(jnt_fld+'aperc_Олимпиада_Кенгуру.jpg')
D = cv2.imread(jnt_fld+'aperc_ПМС_Математика_за_Всеки.jpg')
E = cv2.imread(jnt_fld+'aperc_ПМС_Кенгуру.jpg')
F = cv2.imread(jnt_fld+'aperc_Математика_за_Всеки_Кенгуру.jpg')
row1 = cv2.hconcat([A,B])
row2 = cv2.hconcat([C,D])
row3 = cv2.hconcat([E,F])
im = cv2.vconcat([row1,row2])
img = cv2.vconcat([im,row3])
cv2.imwrite(plt_art_fld + 'joint_distributions.jpg', img)

A = cv2.imread(jnt_fld+'aperc_Олимпиада_ПМС.jpg')
B = cv2.imread(jnt_fld+'aperc_Олимпиада_Математика_за_Всеки.jpg')
C = cv2.imread(jnt_fld+'aperc_Олимпиада_Трансформирано_Кенгуру.jpg')
D = cv2.imread(jnt_fld+'aperc_ПМС_Математика_за_Всеки.jpg')
E = cv2.imread(jnt_fld+'aperc_ПМС_Трансформирано_Кенгуру.jpg')
F = cv2.imread(jnt_fld+'aperc_Математика_за_Всеки_Трансформирано_Кенгуру.jpg')
row1 = cv2.hconcat([A,B])
row2 = cv2.hconcat([C,D])
row3 = cv2.hconcat([E,F])
im = cv2.vconcat([row1,row2])
img = cv2.vconcat([im,row3])
cv2.imwrite(plt_art_fld + 'joint_transformed_distributions.jpg', img)

A = cv2.imread(plot_fld+'by_acceptance/aperc_sof_vs_all_ПМС_Математика_за_Всеки.jpg')
B = cv2.imread(plot_fld+'by_acceptance/aperc_big4_vs_all_ПМС_Математика_за_Всеки_miss.jpg')
C = cv2.imread(plot_fld+'by_acceptance/aperc_sof_vs_all_Олимпиада_Кенгуру.jpg')
D = cv2.imread(plot_fld+'by_acceptance/aperc_big4_vs_all_Олимпиада_Кенгуру_miss.jpg')
row1 = cv2.hconcat([A,B])
row2 = cv2.hconcat([C,D])
img = cv2.vconcat([row1,row2])
cv2.imwrite(plt_art_fld + 'acc_distributions.jpg', img)

A = cv2.imread(plot_fld+'by_region/aperc_Пазарджик_ПМС_Математика_за_Всеки.jpg')
B = cv2.imread(plot_fld+'by_region/aperc_София-град_ПМС_Математика_за_Всеки.jpg')
C = cv2.imread(plot_fld+'by_region/aperc_Пазарджик_Олимпиада_Кенгуру.jpg')
D = cv2.imread(plot_fld+'by_region/aperc_София-град_Олимпиада_Кенгуру.jpg')
row1 = cv2.hconcat([A,B])
row2 = cv2.hconcat([C,D])
img = cv2.vconcat([row1,row2])
cv2.imwrite(plt_art_fld + 'mun_example.jpg', img)

A = cv2.imread(plot_fld+'joint_3d/aperc_sof_vs_all_ПМС_Математика_за_Всеки_Трансформирано_Кенгуру.jpg')
B = cv2.imread(plot_fld+'joint_3d/aperc_big4_vs_all_ПМС_Математика_за_Всеки_Трансформирано_Кенгуру_miss.jpg')
img = cv2.hconcat([A,B])
cv2.imwrite(plt_art_fld + 'triple_comparison.jpg', img)

A = cv2.imread(plot_fld+'hist/0.0_hist_25_SGI.jpg')
B = cv2.imread(plot_fld+'hist/0.2_hist_20_SGI.jpg')
C = cv2.imread(plot_fld+'hist/0.4_hist_15_SGI.jpg')
D = cv2.imread(plot_fld+'hist/0.5_hist_10_SGI.jpg')
row1 = cv2.hconcat([A,B])
row2 = cv2.hconcat([C,D])
img = cv2.vconcat([row1,row2])
cv2.imwrite(plt_art_fld + 'hist_example.jpg', img)