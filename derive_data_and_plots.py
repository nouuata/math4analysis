import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import os
from glob import glob
import cv2
from matplotlib.patches import Patch

generate_data = False
generate_plots = True

file = 'data/merged_ALL_outer.csv'

plot_fld = 'plots/'

fld = 'data/'
if not os.path.exists(fld):
    os.makedirs(fld)
fld = 'data/derived/'
if not os.path.exists(fld):
    os.makedirs(fld)
mfld = fld + 'by_region/'
if not os.path.exists(mfld):
    os.makedirs(mfld)
cdffld = fld + 'cdf/'
if not os.path.exists(cdffld):
    os.makedirs(cdffld)
clsfld = fld + 'classes/'
if not os.path.exists(clsfld):
    os.makedirs(clsfld)
if not os.path.exists(plot_fld):
    os.makedirs(plot_fld)
plt_hist_fld = plot_fld + 'hist/'
if not os.path.exists(plt_hist_fld):
    os.makedirs(plt_hist_fld)

def scatter_sz_plot(x, y, sz, xlabel, ylabel, data_type, folder):
    if not os.path.exists(plot_fld):
        os.makedirs(plot_fld)
    if not os.path.exists(plot_fld+folder):
        os.makedirs(plot_fld+folder)
    plt.clf()
    plt.close()
    px = 1/plt.rcParams['figure.dpi']
    plt.figure(figsize=(1080*px,1080*px))
    plt.scatter(x, y, s=sz)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.gca().invert_xaxis()
    plt.savefig(plot_fld + folder + data_type+"_"+xlabel+"_"+ylabel+".jpg")

def scatter_plot(x, y, xlabel, ylabel, data_type, folder, ext = ''):
    if not os.path.exists(plot_fld):
        os.makedirs(plot_fld)
    if not os.path.exists(plot_fld+folder):
        os.makedirs(plot_fld+folder)
    plt.clf()
    plt.close()
    px = 1/plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(1080*px,1080*px))
    ax = fig.add_subplot(facecolor='whitesmoke')
    ax.scatter(x, y, s=1)
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(plot_fld+folder+data_type+"_"+xlabel+"_"+ylabel+ext+".jpg")

def scatter_plot_3d(x, y, z, xlabel, ylabel, zlabel, data_type, folder = 'joint_3d'):
    if not os.path.exists(plot_fld):
        os.makedirs(plot_fld)
    if not os.path.exists(plot_fld+folder):
        os.makedirs(plot_fld+folder)
    plt.clf()
    plt.close()
    px = 1/plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(1080*px,1080*px))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, s=1)
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.savefig(plot_fld+folder+"/"+data_type+"_"+xlabel+"_"+ylabel+"_"+zlabel+".jpg")

def scatter_size_plot(x, y, sz, xlabel, ylabel, data_type, folder = 'pics_size'):
    if not os.path.exists(plot_fld):
        os.makedirs(plot_fld)
    if not os.path.exists(plot_fld+folder):
        os.makedirs(plot_fld+folder)
    plt.clf()
    plt.close()
    px = 1/plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(1080*px,1080*px))
    ax = fig.add_subplot(facecolor='whitesmoke')
    ax.scatter(x, y, s=sz)
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(plot_fld+folder+"/"+data_type+"_"+xlabel+"_"+ylabel+".jpg")

def scatter_color_plot(x, y, x1, y1, xlabel, ylabel, data_type, name, folder = 'pics_color'):
    if not os.path.exists(plot_fld):
        os.makedirs(plot_fld)
    if not os.path.exists(plot_fld+"pics_color/"):
        os.makedirs(plot_fld+"pics_color/")
    plt.clf()
    plt.close()
    px = 1/plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(1080*px,1080*px))
    ax = fig.add_subplot(facecolor='whitesmoke')
    ax.scatter(x, y, s=1, c='darkorange')
    ax.scatter(x1, y1, s=1, c='red')
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(plot_fld+folder+"/"+data_type+"_"+name+"_"+xlabel+"_"+ylabel+".jpg")

def scatter_color_plot_3d(x, y, z, x1, y1, z1, xlabel, ylabel, zlabel, data_type, folder = 'joint_3d'):
    if not os.path.exists(plot_fld):
        os.makedirs(plot_fld)
    if not os.path.exists(plot_fld+"joint_3d/"):
        os.makedirs(plot_fld+"joint_3d/")
    plt.clf()
    plt.close()
    px = 1/plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(1080*px,1080*px))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, s=1, c='darkorange')
    ax.scatter(x1, y1, z1, s=1, c='red')
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.savefig(plot_fld+folder+"/"+data_type+"_"+xlabel+"_"+ylabel+"_"+zlabel+".jpg")

def scatter_double_color_plot(x, y, x1, y1, x2, y2, xlabel, ylabel, data_type, name, l1, l2, l3, folder = 'pics_color', cl = 'darkorange'):
    if not os.path.exists(plot_fld):
        os.makedirs(plot_fld)
    if not os.path.exists(plot_fld+folder):
        os.makedirs(plot_fld+folder)
    plt.clf()
    plt.close()
    px = 1/plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(1080*px,1080*px))
    ax = fig.add_subplot(facecolor='whitesmoke')
    ax.scatter(x, y, s=1, c=cl, label=l1)
    ax.scatter(x1, y1, s=1, c='red', label=l2)
    ax.scatter(x2, y2, s=1, c='darkgreen', label=l3)
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.savefig(plot_fld+folder+"/"+data_type+"_"+name+"_"+xlabel+"_"+ylabel+".jpg")

def scatter_double_color_plot_3d(x, y, z, x1, y1, z1, x2, y2, z2, xlabel, ylabel, zlabel, data_type, name, l1, l2, l3, folder = 'pics_color'):
    if not os.path.exists(plot_fld):
        os.makedirs(plot_fld)
    if not os.path.exists(plot_fld+folder):
        os.makedirs(plot_fld+folder)
    plt.clf()
    plt.close()
    px = 1/plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(1080*px,1080*px))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, s=1, c='darkorange', label=l1)
    ax.scatter(x1, y1, z1, s=1, c='red', label=l2)
    ax.scatter(x2, y2, z2, s=1, c='darkgreen', label=l3)
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.legend()
    plt.savefig(plot_fld+folder+"/"+data_type+"_"+name+"_"+xlabel+"_"+ylabel+"_"+zlabel+".jpg")

def scatter_triple_color_plot(x, y, x1, y1, x2, y2, x3, y3, xlabel, ylabel, data_type, name, l1, l2, l3, l4, folder = 'pics_color', post = ''):
    if not os.path.exists(plot_fld):
        os.makedirs(plot_fld)
    if not os.path.exists(plot_fld+folder):
        os.makedirs(plot_fld+folder)
    plt.clf()
    plt.close()
    px = 1/plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(1080*px,1080*px))
    ax = fig.add_subplot(facecolor='whitesmoke')
    ax.scatter(x, y, s=1, c='darkorange', label=l1)
    ax.scatter(x1, y1, s=1, c='red', label=l2)
    ax.scatter(x2, y2, s=1, c='darkgreen', label=l3)
    ax.scatter(x3, y3, s=1, c='blue', label=l4)
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.savefig(plot_fld+folder+"/"+data_type+"_"+name+"_"+xlabel+"_"+ylabel + post+".jpg")

def scatter_triple_color_plot_3d(x, y, z, x1, y1, z1, x2, y2, z2, x3, y3, z3, xlabel, ylabel, zlabel, data_type, name, l1, l2, l3, l4, folder = 'pics_color', post = ''):
    if not os.path.exists(plot_fld):
        os.makedirs(plot_fld)
    if not os.path.exists(plot_fld+folder):
        os.makedirs(plot_fld+folder)
    plt.clf()
    plt.close()
    px = 1/plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(1080*px,1080*px))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, s=1, c='darkorange', label=l1)
    ax.scatter(x1, y1, z1, s=1, c='red', label=l2)
    ax.scatter(x2, y2, z2, s=1, c='darkgreen', label=l3)
    ax.scatter(x3, y3, z3, s=1, c='blue', label=l4)
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.legend()
    plt.savefig(plot_fld+folder+"/"+data_type+"_"+name+"_"+xlabel+"_"+ylabel+"_"+zlabel + post+".jpg")

if generate_data:
    df = pd.read_csv(file)

    df['aperc_olymp'] = df['Резултат_Олимпиада'].where(df['Резултат_Олимпиада'].isna(), df['Резултат_Олимпиада'] / 21)
    df['aperc_pms'] = df['Резултат_ПМС'].where(df['Резултат_ПМС'].isna(), df['Резултат_ПМС'] / 49)
    df['aperc_matvs'] = df['Резултат_МатВс'].where(df['Резултат_МатВс'].isna(), df['Резултат_МатВс'] / 42.5)
    df['aperc_keng'] = df['Резултат_Кенг'].where(df['Резултат_Кенг'].isna(), df['Резултат_Кенг'] / 100)
    df['trans_keng'] = df['aperc_keng'].apply(lambda x: x * x)
    df['trans_olymp'] = df['aperc_olymp'] * 0.01
    df['trans_pms'] = df['Резултат_ПМС'].where(df['Резултат_ПМС'].isna(), df['Резултат_ПМС'].clip(None, 38) / 38)

    print("mean aperc_olymp: ", df['aperc_olymp'].mean())
    print("mean aperc_pms: ", df['aperc_pms'].mean())
    print("mean aperc_matvs: ", df['aperc_matvs'].mean())
    print("mean aperc_keng: ", df['aperc_keng'].mean())
    print("mean trans_keng: ", df['trans_keng'].mean())
    print("mean trans_pms: ", df['trans_pms'].mean())

    temp_df = pd.DataFrame()
    temp_df['ol'] = df['trans_olymp']
    temp_df['pm'] = df['aperc_pms']
    temp_df['ma'] = df['aperc_matvs']
    temp_df['ke'] = df['trans_keng']
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)

    wghts = [1,1,0.01,0.01]
    ww = [1,1,1,0.01]
    wghts_sum = 0
    for yy in range(len(wghts)):
        wghts_sum += wghts[yy] * ww[yy]
    print("wghts_sum: ", wghts_sum)
    df['GS'] = temp_df['ol'] + temp_df['pm'] + temp_df['ma'] + temp_df['ke']
    df['GS_rank'] = df['GS'].rank(ascending=False)
    df['GI'] = wghts[0] * temp_df['sc'].str[3] + wghts[1] * temp_df['sc'].str[2] + wghts[2] * temp_df['sc'].str[1] + wghts[3] * temp_df['sc'].str[0]
    df['GI_rank'] = df['GI'].rank(ascending=False)
    df['SGI'] = df['GI'] / wghts_sum
    df['AGI'] = np.exp(df['SGI'] - 1)

    df['comp_count'] = df['Резултат_Олимпиада'].notnull().astype('int') + df['Резултат_ПМС'].notnull().astype('int') + df['Резултат_МатВс'].notnull().astype('int') + df['Резултат_Кенг'].notnull().astype('int')

    mn = 'Благоевград'
    col_mn = 'BL'
    thr = 52
    coef_o, coef_p, coef_m, coef_k = 63,70,63,20
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = df8[['Резултат_Олимпиада', 'Резултат_ПМС', 'Резултат_МатВс', 'Резултат_Кенг']].max()
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 100/mn_o_max, 100/mn_p_max, 100/mn_m_max, 100/mn_k_max
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    mn = 'Бургас'
    col_mn = 'BR'
    thr = 78
    coef_o, coef_p, coef_m, coef_k = 0,1,1,0
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 100/21, 2, 100/45, 1
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    mn = 'Добрич'
    col_mn = 'DB'
    thr = 26
    coef_o, coef_p, coef_m, coef_k = 6,6,6,3
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 100/21, 2, 100/45, 1
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    mn = 'Габрово'
    col_mn = 'GB'
    thr = 26
    coef_o, coef_p, coef_m, coef_k = 2,6,6,0
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 100/21, 2, 100/45, 1
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    mn = 'Хасково'
    col_mn = 'HS'
    thr = 52
    coef_o, coef_p, coef_m, coef_k = 100,100,100,100
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = df[['Резултат_Олимпиада', 'Резултат_ПМС', 'Резултат_МатВс', 'Резултат_Кенг']].max()
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 100/mn_o_max, 100/mn_p_max, 100/mn_m_max, 100/mn_k_max
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    df['KR'] = 0

    mn = 'Кюстендил'
    col_mn = 'KS'
    thr = 52
    coef_o, coef_p, coef_m, coef_k = 19,8,9,4
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 1,1,1,1
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    mn = 'Ловеч'
    col_mn = 'LV'
    thr = 26
    coef_o, coef_p, coef_m, coef_k = 1,0,2,1
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 100/21, 2, 100/45, 1
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    mn = 'Монтана'
    col_mn = 'MN'
    thr = 26
    coef_o, coef_p, coef_m, coef_k = 5,2,2,1
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 1,1,1,1
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    mn = 'Плевен'
    col_mn = 'PL'
    thr = 26
    coef_o, coef_p, coef_m, coef_k = 3,2,2,1
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 1,1,1,1
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    mn = 'Перник'
    col_mn = 'PK'
    thr = 26
    coef_o, coef_p, coef_m, coef_k = 8,3,4,1
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 1,1,1,1
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    mn = 'Пловдив'
    col_mn = 'PV'
    thr = 52
    coef_o, coef_p, coef_m, coef_k = 4,4,8,0
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = df8[['Резултат_Олимпиада', 'Резултат_ПМС', 'Резултат_МатВс', 'Резултат_Кенг']].max()
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 100/mn_o_max, 100/mn_p_max, 100/mn_m_max, 100/mn_k_max
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    mn = 'Пазарджик'
    col_mn = 'PZ'
    thr = 52
    coef_o, coef_p, coef_m, coef_k = 1,1,1,1
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 100/21, 2, 100/45, 1
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    mn = 'Русе'
    col_mn = 'RS'
    thr = 52
    coef_o, coef_p, coef_m, coef_k = 17, 7, 8, 0
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 1,1,1,1
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    mn = 'Разград'
    col_mn = 'RZ'
    thr = 26
    coef_o, coef_p, coef_m, coef_k = 1,1,1,1
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 100/21,2,100/45,1
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    mn = 'София-град'
    col_mn = 'SF'
    smg_o_min, smg_p_min, smg_m_min = 1, 0, 2.5
    smg_o_max, smg_p_max, smg_m_max = 21, 49, 40
    npmg_o_max, npmg_p_max, npmg_m_max = 21, 46.5, 38.5
    smg_oly = 100*((df['Резултат_Олимпиада'] - smg_o_min + 1)/(smg_o_max - smg_o_min + 1))
    smg_pms = 100*((df['Резултат_ПМС'] - smg_p_min + 1)/(smg_p_max - smg_p_min + 1))
    smg_mat = 100*((df['Резултат_МатВс'] - smg_m_min + 1)/(smg_m_max - smg_m_min + 1))
    smg_oly = smg_oly.fillna(value=0)
    smg_pms = smg_pms.fillna(value=0)
    smg_mat = smg_mat.fillna(value=0)
    npmg_oly = 5*100*(df['Резултат_Олимпиада']/npmg_o_max)
    npmg_pms = 6*100*(df['Резултат_ПМС']/npmg_p_max)
    npmg_mat = 8*100*(df['Резултат_МатВс']/npmg_m_max)
    npmg_oly = npmg_oly.fillna(value=0)
    npmg_pms = npmg_pms.fillna(value=0)
    npmg_mat = npmg_mat.fillna(value=0)
    min_all = np.minimum(npmg_oly, npmg_pms)
    min_all = np.minimum(min_all, npmg_mat)
    df[col_mn +'_smg'] = (1.5 * smg_oly + smg_pms) / 2.5 + 6 * smg_mat + 150
    df[col_mn +'_npmg'] = npmg_oly + npmg_pms + npmg_mat - min_all + 150
    # na treto klasirane v smg e obqven minimalen bal 519.99
    # na treto klasirane v npmg e obqven minimalen bal 1021.49
    df[col_mn] = (df['Област'] == mn) & ((df[col_mn +'_smg'] >= 519.99) | (df[col_mn +'_npmg'] >= 1021.49))
    df[col_mn] = df[col_mn].astype('int')

    mn = 'Шумен'
    col_mn = 'SH'
    thr = 52
    coef_o, coef_p, coef_m, coef_k = 1,1,1,1
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 100/21,2,100/45,1
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    mn = 'Сливен'
    col_mn = 'SL'
    thr = 52
    coef_o, coef_p, coef_m, coef_k = 3,3,3,2
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 100/21,2,100/45,1
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    mn = 'Смолян'
    col_mn = 'SM'
    thr = 26
    coef_o, coef_p, coef_m, coef_k = 1,1,1,1
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 100/21,2,100/45,1
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    mn = 'София област'
    col_mn = 'SO'
    thr = 52
    coef_o, coef_p, coef_m, coef_k = 5,0,2,1
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 1,1,1,1
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    mn = 'Силистра'
    col_mn = 'SS'
    thr = 26
    coef_o, coef_p, coef_m, coef_k = 1,1,1,1
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 100/21,2,100/45,1
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    mn = 'Стара Загора'
    col_mn = 'SZ'
    thr = 104
    coef_o, coef_p, coef_m, coef_k = 1,1,0,1
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 100/21,2,100/45,1
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    df['TR'] = 0

    mn = 'Враца'
    col_mn = 'VC'
    thr = 52
    coef_o, coef_p, coef_m, coef_k = 1,1,1,1
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 100/21,2,100/45,1
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    mn = 'Видин'
    col_mn = 'VD'
    thr = 26
    coef_o, coef_p, coef_m, coef_k = 6,6,6,3
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 100/21,2,100/45,1
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    mn = 'Варна'
    col_mn = 'VR'
    thr = 78
    coef_o, coef_p, coef_m, coef_k = 10,12,15,0
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = df8[['Резултат_Олимпиада', 'Резултат_ПМС', 'Резултат_МатВс', 'Резултат_Кенг']].max()
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 100/mn_o_max, 100/mn_p_max, 100/mn_m_max, 100/mn_k_max
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    df['VT'] = 0

    mn = 'Ямбол'
    col_mn = 'YM'
    thr = 52
    coef_o, coef_p, coef_m, coef_k = 1,1,1,0
    df8 = df[df['Област'] == mn]
    mn_o_max, mn_p_max, mn_m_max, mn_k_max = 100/21,2,100/45,1
    temp_df = pd.DataFrame()
    temp_df['ol'] = coef_o*(df['Резултат_Олимпиада']*mn_o_max)
    temp_df['pm'] = coef_p*(df['Резултат_ПМС']*mn_p_max)
    temp_df['ma'] = coef_m*(df['Резултат_МатВс']*mn_m_max)
    temp_df['ke'] = coef_k*(df['Резултат_Кенг']*mn_k_max)
    temp_df = temp_df.fillna(value=0)
    temp_df['sc'] = temp_df.values.tolist()
    temp_df['sc'] = temp_df['sc'].apply(sorted)
    df[col_mn+'_mg'] = temp_df['sc'].str[2] + temp_df['sc'].str[3]
    df[col_mn] = df[df['Област'] == mn][col_mn+'_mg'].rank(ascending=False) <= thr
    df[[col_mn]] = df[[col_mn]].fillna(value=0)
    df[col_mn] = df[col_mn].astype('int')

    df['badm'] = df['BL'] + df['BR'] + df['DB'] + df['GB'] + df['HS'] + df['KS'] + df['LV'] + df['MN'] + df['PL'] + df['PK'] + df['PV'] + df['PZ'] + df['RS'] + df['RZ'] + df['SF']+ df['SH'] + df['SL'] + df['SM']+ df['SO'] + df['SS'] + df['SZ'] + df['VC']+ df['VD'] + df['VR'] + df['YM']
    df['top1adm'] = df['BL'] + df['BR'] + df['DB'] + df['GB'] + df['HS'] + df['KS'] + df['LV'] + df['MN'] + df['PL'] + df['PK'] + df['PV'] + df['PZ'] + df['RS'] + df['RZ'] + 2*df['SF']+ df['SH'] + df['SL'] + df['SM']+ df['SO'] + df['SS'] + df['SZ'] + df['VC']+ df['VD'] + df['VR'] + df['YM']
    df['top4adm'] = df['BL'] + 2*df['BR'] + df['DB'] + df['GB'] + df['HS'] + df['KS'] + df['LV'] + df['MN'] + df['PL'] + df['PK'] + 2*df['PV'] + df['PZ'] + df['RS'] + df['RZ'] + 2*df['SF']+ df['SH'] + df['SL'] + df['SM']+ df['SO'] + df['SS'] + df['SZ'] + df['VC']+ df['VD'] + 2*df['VR'] + df['YM']

    df['top4'] = (df['Област'] == 'София-град') | (df['Област'] == 'Варна') | (df['Област'] == 'Бургас') | (df['Област'] == 'Пловдив')
    df['top1'] = df['Област'] == 'София-град'
    df['top4'] = df['top4'].astype('int')
    df['top1'] = df['top1'].astype('int')

    column_to_move = df.pop('GS')
    df.insert(8, 'GS', column_to_move)
    column_to_move = df.pop('GS_rank')
    df.insert(9, 'GS_rank', column_to_move)
    column_to_move = df.pop('GI_rank')
    df.insert(10, 'GI_rank', column_to_move)
    column_to_move = df.pop('GI')
    df.insert(11, 'GI', column_to_move)
    column_to_move = df.pop('badm')
    df.insert(12, 'badm', column_to_move)
    column_to_move = df.pop('SGI')
    df.insert(13, 'SGI', column_to_move)
    column_to_move = df.pop('AGI')
    df.insert(14, 'AGI', column_to_move)

    df = df.sort_values(by='GI', ascending=False)

    dfx = df[df['badm'] == 0]
    dfx['F_rank'] = dfx['GI'].rank(ascending=False)

    dfs = dfx[dfx['Област'] == 'София-град']
    dfs['S_rank'] = dfs['GI'].rank(ascending=False)

    dft = dfx[dfx['top4'] == 1]
    dft['P_rank'] = dft['GI'].rank(ascending=False)

    th_value = dfs.loc[dfs['S_rank'] == 52]['SGI'].iloc[0]
    dfp = df[(df['badm'] == 1) & (df['top4'] == 0 ) & (df['SGI'] < th_value)]
    print("SF 52 SGI: ", th_value, " : ", dfp.shape)

    th_value = dft.loc[dft['P_rank'] == 100]['SGI'].iloc[0]
    dfp = df[(df['badm'] == 1) & (df['top4'] == 0 ) & (df['SGI'] < th_value)]
    print("TOP4 100 SGI: ", th_value, " : ", dfp.shape)

    th_value = dfx.loc[dfx['F_rank'] == 200]['SGI'].iloc[0]
    dfp = df[(df['badm'] == 1) & (df['top4'] == 0 ) & (df['SGI'] < th_value)]
    print("TOP4 200 SGI: ", th_value, " : ", dfp.shape)

    th_value = dfx.loc[dfx['F_rank'] == 400]['SGI'].iloc[0]
    dfp = df[(df['badm'] == 1) & (df['top4'] == 0 ) & (df['SGI'] < th_value)]
    print("TOP4 400 SGI: ", th_value, " : ", dfp.shape)

    th_value = dfx.loc[dfx['F_rank'] == 800]['SGI'].iloc[0]
    dfp = df[(df['badm'] == 1) & (df['top4'] == 0 ) & (df['SGI'] < th_value)]
    print("TOP4 800 SGI: ", th_value, " : ", dfp.shape)

    df.to_csv(fld+'data_ALL.csv', index=False, quoting=csv.QUOTE_NONE)

    df44 = df[(df['GI_rank'] <= 1300) & (df['badm'] == 0)]
    print("lost top @ 1300 :", df44.shape[0])
    df44 = df[(df['GI_rank'] <= 1300)]
    df44.to_csv(fld+'top1300.csv', index=False, quoting=csv.QUOTE_NONE)

    df45 = df[(df['GI_rank'] <= 1300) & (df['top4'] == 1)]
    print("big4 top @ 1300 :", df45.shape[0])

    df47 = df[(df['Област'] == 'София-град') & (df['badm'] == 1)]
    print(df47.shape)
    df48 = df47.groupby(['Училище'])['Училище'].agg('count').to_frame('count').reset_index()
    df48.to_csv(fld+'sof_adm.csv', index=False, quoting=csv.QUOTE_NONE)

    mun = ['Благоевград','Бургас','Добрич','Габрово','Хасково','Кърджали','Кюстендил','Ловеч','Монтана','Плевен','Перник','Пловдив','Пазарджик','Русе','Разград','София-град','Шумен','Сливен','Смолян','София област','Силистра','Стара Загора','Търговище','Враца','Видин','Варна','Велико Търново','Ямбол']
    mun_code = ['BL','BR','DB','GB','HS','KR','KS','LV','MN','PL','PK','PV','PZ','RS','RZ','SF','SH','SL','SM','SO','SS','SZ','TR','VC','VD','VR','VT','YM']

    for mi in range(len(mun)):
        mn = mun[mi]
        mn_code = mun_code[mi]
        df11 = df[df['Област'] == mn]
        df11 = df11.sort_values(by='GI', ascending=False)
        df11.to_csv(mfld+'all_' + mn_code + '.csv', index=False, quoting=csv.QUOTE_NONE)

    mun_lst = ['Благоевград','Бургас','Варна','Велико Търново','Видин','Враца','Габрово','Добрич','Кърджали','Кюстендил','Ловеч','Монтана','Пазарджик','Перник','Плевен','Пловдив','Разград','Русе','Силистра','Сливен','Смолян','София област','София-град','Стара Загора','Търговище','Хасково','Шумен','Ямбол']
    pr_lst = [52,78,78,0, 26,52,26,26, 0,52,26,26, 52,26,26,52, 26,52,26,52, 26,52,208,104,0,52,52,52]
    cl_lst = [2,3,3,0, 1,2,1,1, 0,2,1,1, 2,1,1,2, 1,2,1,2, 1,2,8,4,0,2,2,2]
    d = {'Област': mun_lst, 'Прием': pr_lst, 'Класове' : cl_lst}
    df13 = pd.DataFrame(data=d)

    thres = 1300
    df12 = df[df['GI_rank'] <= thres]
    df14 = df12.groupby(['Област'])['Област'].agg('count').to_frame('count').reset_index()
    df15 = pd.merge(df13, df14, on='Област', how='outer')
    df15 = df15.fillna(value=0)
    df15['diff'] = df15['Прием'] - df15['count']
    df15['classes'] = df15['diff'] / 26
    df15['classes'] = df15['classes'].round(0).astype(int)
    df15.to_csv(fld+'top'+str(thres)+'_by_mun_count.csv', index=False)

    adm_cl_scores = list()
    real_adm_cl_scores = list()
    for mi in range(len(mun_lst)):
        mn = mun_lst[mi]
        df16 = df[(df['Област'] == mn) & (df['badm'] == 1)]
        sts = pr_lst[mi]
        clss = cl_lst[mi]
        gsc_lst = (df16['SGI'] -1).to_list()
        cls_sc = sum(gsc_lst[0:clss*26])/26
        top_cls_sc = 0
        low_cls_sc = 0
        sc_cls_sc = cls_sc
        if clss > 0:
            top_cls_sc = math.exp(sum(gsc_lst[0:26])/26)
            low_cls_sc = math.exp(sum(gsc_lst[-26:])/26)
            sc_cls_sc = math.exp(cls_sc / clss)
            real_adm_cl_scores.append(sc_cls_sc)
        tlst = [mn, top_cls_sc, low_cls_sc, sc_cls_sc]
        adm_cl_scores.append(tlst)

    min_adm_sc = min(real_adm_cl_scores)
    min_adm_scores = [0.0, min_adm_sc, 0.45, 0.475, 0.5]
    new_clss = list()
    for mas in min_adm_scores:
        this_clss = [0]*len(mun_lst)
        class_lst = list()
        for mi in range(len(mun_lst)):
            mn = mun_lst[mi]
            df16 = df[(df['Област'] == mn) & (df['badm'] == 1)]
            sts = pr_lst[mi]
            clss = cl_lst[mi]
            gsc_lst = (df16['SGI'] -1).to_list()
            for cc in range(clss):
                cls_sc = math.exp(sum(gsc_lst[cc*26:(cc+1)*26]) / 26)
                tlst = [mn, 1, math.exp(gsc_lst[cc*26]), cls_sc]
                class_lst.append(tlst)
        for mi in range(len(mun_lst)):
            mn = mun_lst[mi]
            df16 = df[(df['Област'] == mn) & (df['badm'] == 0)]
            sts = df16.shape[0]
            clss = math.floor(sts/26)
            sts = clss * 26
            gsc_lst = (df16['SGI'] -1).to_list()
            for cc in range(clss):
                cls_sc = math.exp(sum(gsc_lst[cc*26:(cc+1)*26])/26)
                if cls_sc < mas:
                    break
                tlst = [mn, 0, math.exp(gsc_lst[cc*26]), cls_sc]
                class_lst.append(tlst)
                this_clss[mi] = this_clss[mi] + 1
        top_cl_df = pd.DataFrame(class_lst, columns = ['Област', 'badm', 'top_student', 'class_score'])
        top_cl_df.to_csv(clsfld+'top_classes_'+str(mas)+'.csv', index=False)
        this_clss.insert(0, mas)
        new_clss.append(this_clss)
        if mas == 0.0:
            top_cl_df['class_rank'] = top_cl_df['class_score'].rank(ascending=False)
            top_cl_df.to_csv(clsfld+'classes_rank.csv', index=False)
            top_cl_df = top_cl_df[top_cl_df['class_rank'] <= 50]
            df31 = top_cl_df.groupby(['Област'])['Област'].agg('count').to_frame('count').reset_index()
            df32 = pd.merge(df13, df31, on='Област', how='outer')
            df32 = df32.fillna(0)
            df32.to_csv(clsfld+'top50_classes.csv', index=False)

    dd = ['thresh']
    dd.extend(mun_lst)
    new_cl_df = pd.DataFrame(new_clss, columns = dd)
    new_cl_df.to_csv(clsfld+'new_classes_by_mun.csv', index=False)

    classes_by_mun_df = pd.DataFrame(adm_cl_scores, columns = ['Област', 'top_adm_cl_scores', 'low_adm_cl_scores', 'adm_ave_class_sc'])
    classes_by_mun_df.to_csv(clsfld+'classes_by_mun.csv', index=False)

    d = {'Област': mun_lst, 'adm_ave_class_sc' : adm_cl_scores}
    df21 = pd.DataFrame(data=d)

    varArr1 = ['Резултат_Олимпиада', 'Резултат_ПМС', 'Резултат_МатВс', 'Резултат_Кенг', 'trans_keng', 'GI']
    varArr2 = ['olymp', 'pms', 'matvs', 'keng', 'trans_keng', 'GI']

    for vi in range(len(varArr1)):
        var1 = varArr1[vi]
        var2 = varArr2[vi]
        df7 = df.groupby([var1])[var1].agg('count').to_frame('count').reset_index()
        df7['cumsum'] = df7['count'].cumsum()
        df7.to_csv(cdffld+var2+'_count.csv', index=False)

#///////////////////////////////////////////////////////
if not generate_plots:
    exit()

df100 = pd.read_csv(clsfld+'classes_rank.csv')
df100 = df100.sort_values(by='class_rank', ascending=False)
barWidth = 0.35
plt.clf()
plt.close()
px = 1/plt.rcParams['figure.dpi']
fig, ax = plt.subplots(1, 1, figsize=(1080*px,7000*px))
acc_cls = df100['class_score'].to_list()
sup_cls = df100['top_student'].to_list()
badm_lst = df100['badm'].to_list()
label_lst = df100['Област'].to_list()
br1 = np.arange(len(acc_cls)) 
br2 = [x + barWidth for x in br1] 
b1list = ax.barh(br1, acc_cls, barWidth, color ='g', 
        edgecolor ='grey', label ='Индекс на приет клас')
b2list = ax.barh(br2, sup_cls, barWidth, color ='lightgreen', 
        edgecolor ='grey', label ='Индекс на най-силен приет ученик')
for bi in range(len(badm_lst)):
    if badm_lst[bi] == 0:
        b1list[bi].set_color('firebrick')
        b2list[bi].set_color('coral')
old_handles, labels = ax.get_legend_handles_labels()
legend_elements = [Patch(facecolor='g', label='Индекс на приет клас'), 
                   Patch(facecolor='lightgreen', label='Индекс на най-силен приет ученик'),
                   Patch(facecolor='firebrick', label='Индекс на неприет клас'), 
                   Patch(facecolor='coral', label='Индекс на най-силен неприет ученик')]
ax.set_xlim([0.368,1.05])
fig.subplots_adjust(left=0.15)
plt.xlabel('AGI', fontweight ='bold', fontsize = 10) 
plt.ylabel('Област', fontweight ='bold', fontsize = 10)
plt.yticks([r + 0.5*barWidth for r in range(len(acc_cls))], label_lst)
plt.legend(handles=legend_elements)
plt.savefig(plot_fld+'dist_AGI_classes.jpg')

plt.clf()
plt.close()
px = 1/plt.rcParams['figure.dpi']
fig, ax = plt.subplots(1, 1, figsize=(1080*px,720*px))
tp = -40
cacc_cls = acc_cls[tp:]
csup_cls = sup_cls[tp:]
cbadm_lst = badm_lst[tp:]
clabel_lst = label_lst[tp:]
br1 = np.arange(len(cacc_cls)) 
br2 = [x + barWidth for x in br1] 
b1list = ax.barh(br1, cacc_cls, barWidth, color ='g', 
        edgecolor ='grey', label ='Индекс на приет клас')
b2list = ax.barh(br2, csup_cls, barWidth, color ='lightgreen', 
        edgecolor ='grey', label ='Индекс на най-силен приет ученик')
for bi in range(len(cbadm_lst)):
    if cbadm_lst[bi] == 0:
        b1list[bi].set_color('firebrick')
        b2list[bi].set_color('coral')
old_handles, labels = ax.get_legend_handles_labels()
legend_elements = [Patch(facecolor='g', label='Индекс на приет клас'), 
                   Patch(facecolor='lightgreen', label='Индекс на най-силен приет ученик'),
                   Patch(facecolor='firebrick', label='Индекс на неприет клас'), 
                   Patch(facecolor='coral', label='Индекс на най-силен неприет ученик')]
ax.set_xlim([0.368,1.05])
fig.subplots_adjust(left=0.15, top = 0.99, bottom = 0.06)
plt.xlabel('AGI', fontweight ='bold', fontsize = 10) 
plt.ylabel('Област', fontweight ='bold', fontsize = 10)
plt.yticks([r + 0.5*barWidth for r in range(len(cacc_cls))], clabel_lst)
plt.legend(handles=legend_elements)
plt.savefig(plot_fld+'dist_AGI_classes_1.jpg')

plt.clf()
plt.close()
px = 1/plt.rcParams['figure.dpi']
fig, ax = plt.subplots(1, 1, figsize=(1080*px,1620*px))
tp = -120
bp = -40
cacc_cls = acc_cls[tp:bp]
csup_cls = sup_cls[tp:bp]
cbadm_lst = badm_lst[tp:bp]
clabel_lst = label_lst[tp:bp]
br1 = np.arange(len(cacc_cls)) 
br2 = [x + barWidth for x in br1] 
b1list = ax.barh(br1, cacc_cls, barWidth, color ='g', 
        edgecolor ='grey', label ='Индекс на приет клас')
b2list = ax.barh(br2, csup_cls, barWidth, color ='lightgreen', 
        edgecolor ='grey', label ='Индекс на най-силен приет ученик')
for bi in range(len(cbadm_lst)):
    if cbadm_lst[bi] == 0:
        b1list[bi].set_color('firebrick')
        b2list[bi].set_color('coral')
old_handles, labels = ax.get_legend_handles_labels()
legend_elements = [Patch(facecolor='g', label='Индекс на приет клас'), 
                   Patch(facecolor='lightgreen', label='Индекс на най-силен приет ученик'),
                   Patch(facecolor='firebrick', label='Индекс на неприет клас'), 
                   Patch(facecolor='coral', label='Индекс на най-силен неприет ученик')]
ax.set_xlim([0.368,1.05])
fig.subplots_adjust(left=0.15, top = 0.99, bottom = 0.06)
plt.xlabel('AGI', fontweight ='bold', fontsize = 10) 
plt.ylabel('Област', fontweight ='bold', fontsize = 10)
plt.yticks([r + 0.5*barWidth for r in range(len(cacc_cls))], clabel_lst)
plt.legend(handles=legend_elements)
plt.savefig(plot_fld+'dist_AGI_classes_2.jpg')

plt.clf()
plt.close()
px = 1/plt.rcParams['figure.dpi']
fig, ax = plt.subplots(1, 1, figsize=(1080*px,1620*px))
tp = -201
bp = -120
cacc_cls = acc_cls[tp:bp]
csup_cls = sup_cls[tp:bp]
cbadm_lst = badm_lst[tp:bp]
clabel_lst = label_lst[tp:bp]
br1 = np.arange(len(cacc_cls)) 
br2 = [x + barWidth for x in br1] 
b1list = ax.barh(br1, cacc_cls, barWidth, color ='g', 
        edgecolor ='grey', label ='Индекс на приет клас')
b2list = ax.barh(br2, csup_cls, barWidth, color ='lightgreen', 
        edgecolor ='grey', label ='Индекс на най-силен приет ученик')
for bi in range(len(cbadm_lst)):
    if cbadm_lst[bi] == 0:
        b1list[bi].set_color('firebrick')
        b2list[bi].set_color('coral')
old_handles, labels = ax.get_legend_handles_labels()
legend_elements = [Patch(facecolor='g', label='Индекс на приет клас'), 
                   Patch(facecolor='lightgreen', label='Индекс на най-силен приет ученик'),
                   Patch(facecolor='firebrick', label='Индекс на неприет клас'), 
                   Patch(facecolor='coral', label='Индекс на най-силен неприет ученик')]
ax.set_xlim([0.368,1.05])
fig.subplots_adjust(left=0.15, top = 0.99, bottom = 0.06)
plt.xlabel('AGI', fontweight ='bold', fontsize = 10) 
plt.ylabel('Област', fontweight ='bold', fontsize = 10)
plt.yticks([r + 0.5*barWidth for r in range(len(cacc_cls))], clabel_lst)
plt.legend(handles=legend_elements)
plt.savefig(plot_fld+'dist_AGI_classes_3.jpg')

exit()

rmun_lst = ['Благоевград','Бургас','Варна','Велико Търново','Видин','Враца','Габрово','Добрич','Кърджали','Кюстендил','Ловеч','Монтана','Пазарджик','Перник','Плевен','Пловдив','Разград','Русе','Силистра','Сливен','Смолян','София област','София-град','Стара Загора','Търговище','Хасково','Шумен','Ямбол']
rmun_lst.reverse()

df32 = pd.read_csv(clsfld+'top50_classes.csv')
df32 = df32.sort_values(by='Област', ascending=False)
barWidth = 0.35
plt.clf()
plt.close()
px = 1/plt.rcParams['figure.dpi']
fig, ax = plt.subplots(1, 1, figsize=(1080*px,540*px))
acc_cls = df32['Класове'].to_list()
sup_cls = df32['count'].to_list()
br1 = np.arange(len(acc_cls)) 
br2 = [x + barWidth for x in br1] 
ax.barh(br1, acc_cls, barWidth, color ='r', 
        edgecolor ='grey', label ='Прием на класове') 
ax.barh(br2, sup_cls, barWidth, color ='g', 
        edgecolor ='grey', label ='Kласове по резултат')
fig.subplots_adjust(left=0.15, top = 0.99, bottom = 0.04)
plt.xlabel('Брой класове', fontweight ='bold', fontsize = 10) 
plt.ylabel('Област', fontweight ='bold', fontsize = 10)
plt.yticks([r + 0.5*barWidth for r in range(len(acc_cls))], rmun_lst)
plt.legend()
plt.savefig(plot_fld+'classes.jpg')

df32 = pd.read_csv(clsfld+'classes_by_mun.csv')
df32 = df32.sort_values(by='Област', ascending=False)
barWidth = 0.25
plt.clf()
plt.close()
px = 1/plt.rcParams['figure.dpi']
fig, ax = plt.subplots(1, 1, figsize=(1080*px,540*px))
acc_cls = df32['top_adm_cl_scores'].to_list()
low_cls = df32['low_adm_cl_scores'].to_list()
sup_cls = df32['adm_ave_class_sc'].to_list()
br1 = np.arange(len(acc_cls)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + 2*barWidth for x in br1] 
ax.barh(br1, acc_cls, barWidth, color ='g', 
        edgecolor ='grey', label ='Най-силен клас')
ax.barh(br2, sup_cls, barWidth, color ='gold', 
        edgecolor ='grey', label ='Средно за областта')
ax.barh(br3, low_cls, barWidth, color ='r', 
        edgecolor ='grey', label ='Най-слаб клас')
ax.set_xlim([0.368,1.0])
fig.subplots_adjust(left=0.15, top = 0.99, bottom = 0.04)
plt.xlabel('AGI', fontweight ='bold', fontsize = 10) 
plt.ylabel('Област', fontweight ='bold', fontsize = 10)
plt.yticks([r + barWidth for r in range(len(acc_cls))], rmun_lst)
plt.legend()
plt.savefig(plot_fld+'classes_by_region.jpg')

varArr1 = ['Резултат_Олимпиада', 'Резултат_ПМС', 'Резултат_МатВс', 'Резултат_Кенг', 'trans_keng', 'GI']
varArr2 = ['olymp', 'pms', 'matvs', 'keng', 'trans_keng', 'GI']
varArr3 = ['Резултат_Олимпиада', 'Резултат_ПМС', 'Резултат_Математика_за_Всеки', 'Резултат_Кенгуру', 'Трансформирано_Кенгуру', 'Глобален_индекс']

for vi in range(len(varArr1)):
    var1 = varArr1[vi]
    var2 = varArr2[vi]
    var3 = varArr3[vi]
    df7 = pd.read_csv(cdffld+var2+'_count.csv')
    scatter_sz_plot(df7[var1].to_list(), df7['cumsum'].to_list(), df7['count'].to_list(), var3, 'Брой_участници', 'cdf', 'cdf/')

if not os.path.exists(plot_fld + 'hist/'):
    os.makedirs(plot_fld + 'hist/')

v_th = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
h_bins = [10,15,20,25]
for vth in v_th:
    df7 = df[df['SGI'] >= vth]
    df12 = df7[(df7['Област']=='София-град') | (df7['badm'] == 1)]
    df11 = df7[(df7['Област']=='Бургас') | (df7['Област']=='Варна') | (df7['Област']=='Пловдив') | (df7['Област']=='София-град') | (df7['badm'] == 1)]
    df13 = df7[(df7['Област']=='Велико Търново') | (df7['Област']=='Кърджали') | (df7['Област']=='Търговище') | (df7['Област']=='Бургас') | (df7['Област']=='Варна') | (df7['Област']=='Пловдив') | (df7['Област']=='София-град') | (df7['badm'] == 1)]
    df8 = df7[df7['badm'] == 1]
    df9 = df8[df8['Област']=='София-град']
    df10 = df8[(df8['Област']=='Бургас') | (df8['Област']=='Варна') | (df8['Област']=='Пловдив') | (df8['Област']=='София-град')]
    for gi in range(len(h_bins)):
        bns = h_bins[gi]
        plt.clf()
        plt.close()
        px = 1/plt.rcParams['figure.dpi']
        plt.figure(figsize=(1080*px,1080*px))
        bms = np.histogram(np.hstack((df7.SGI, df13.SGI, df11.SGI, df12.SGI, df8.SGI, df10.SGI, df9.SGI)), bins = bns)[1]
        plt.hist(df7.SGI, bins = bms, color='orange', label ='неприети провинция')
        plt.hist(df13.SGI, bins = bms, color='midnightblue', label ='без прием Велико Търново, Кърджали, Търговище')
        plt.hist(df11.SGI, bins = bms, color='pink', label ='неприети Варна, Бургас, Пловдив')
        plt.hist(df12.SGI, bins = bms, color='purple', label ='неприети София')
        plt.hist(df8.SGI, bins = bms, color='limegreen', label ='приети провинция')
        plt.hist(df10.SGI, bins = bms, color='blue', label ='приети Варна, Бургас, Пловдив')
        plt.hist(df9.SGI, bins = bms, color='red', label ='приети София')
        plt.xlabel('SGI', fontweight ='bold', fontsize = 20) 
        plt.legend()
        plt.savefig(plt_hist_fld+str(vth)+'_hist_'+str(bns)+'_SGI.jpg')

varArr1 = ['Резултат_Олимпиада', 'Резултат_Олимпиада', 'Резултат_Олимпиада', 'Резултат_ПМС', 'Резултат_ПМС', 'Резултат_МатВс','Резултат_Олимпиада', 'Резултат_ПМС', 'Резултат_МатВс']
varArr2 = ['Резултат_ПМС', 'Резултат_МатВс', 'Резултат_Кенг', 'Резултат_МатВс', 'Резултат_Кенг', 'Резултат_Кенг', 'trans_keng', 'trans_keng', 'trans_keng']
shArr1 = ['aperc_olymp', 'aperc_olymp', 'aperc_olymp', 'aperc_pms', 'aperc_pms', 'aperc_matvs', 'aperc_olymp', 'aperc_pms', 'aperc_matvs']
shArr2 = ['aperc_pms', 'aperc_matvs', 'aperc_keng', 'aperc_matvs', 'aperc_keng', 'aperc_keng', 'trans_keng', 'trans_keng', 'trans_keng']
discArr1 = ['Олимпиада', 'Олимпиада', 'Олимпиада', 'ПМС', 'ПМС', 'Математика_за_Всеки', 'Олимпиада', 'ПМС', 'Математика_за_Всеки']
discArr2 = ['ПМС', 'Математика_за_Всеки', 'Кенгуру', 'Математика_за_Всеки', 'Кенгуру', 'Кенгуру', 'Трансформирано_Кенгуру', 'Трансформирано_Кенгуру', 'Трансформирано_Кенгуру']
vv = ['aperc']

df2 = df[df['Резултат_Кенг'].notna()]
scatter_plot(df2['aperc_keng'].to_list(), df2['aperc_keng'].to_list(), "Състезание_1", "Състезание_2", 'aperc', '', '')

for i in range(len(varArr1)):
    for vw in vv:
        var1 = varArr1[i]
        var2 = varArr2[i]
        var3 = shArr1[i]
        var4 = shArr2[i]
        df2 = df[df[var1].notna() & df[var2].notna()]
        df5 = df2.groupby([var3,var4])[var3].agg('count').to_frame('count').reset_index()
        scatter_plot(df2[var3].to_list(), df2[var4].to_list(), discArr1[i], discArr2[i], vw, 'joint_dist/')
        scatter_size_plot(df5[var3].to_list(), df5[var4].to_list(), df5['count'].to_list(), discArr1[i], discArr2[i], "sz_"+vw, "joint_dist/with_size")

varArr1 = ['Резултат_ПМС', 'Резултат_МатВс', 'Резултат_Олимпиада']
for i in range(len(varArr1)):
    var1 = varArr1[i]
    var2 = 'trans_keng'
    df2 = df[df[var1].notna() & df[var2].notna()]
    correl = df2[var1].corr(df2[var2])
    print(var1, " : ", var2, " : correl: ", correl)

varArr1 = ['Резултат_Олимпиада', 'Резултат_Олимпиада', 'Резултат_Олимпиада', 'Резултат_ПМС', 'Резултат_ПМС', 'Резултат_МатВс']
varArr2 = ['Резултат_ПМС', 'Резултат_МатВс', 'Резултат_Кенг', 'Резултат_МатВс', 'Резултат_Кенг', 'Резултат_Кенг']
for i in range(len(varArr1)):
    var1 = varArr1[i]
    var2 = varArr2[i]
    df2 = df[df[var1].notna() & df[var2].notna()]
    correl = df2[var1].corr(df2[var2])
    print(var1, " : ", var2, " : correl: ", correl)

varArr1 = ['Резултат_Олимпиада', 'Резултат_Олимпиада', 'Резултат_Олимпиада', 'Резултат_ПМС', 'Резултат_ПМС', 'Резултат_МатВс']
varArr2 = ['Резултат_ПМС', 'Резултат_МатВс', 'Резултат_Кенг', 'Резултат_МатВс', 'Резултат_Кенг', 'Резултат_Кенг']
shArr1 = ['olymp', 'olymp', 'olymp', 'pms', 'pms', 'matvs']
shArr2 = ['pms', 'matvs', 'keng', 'matvs', 'keng', 'keng']
discArr1 = ['Олимпиада', 'Олимпиада', 'Олимпиада', 'ПМС', 'ПМС', 'Математика_за_Всеки']
discArr2 = ['ПМС', 'Математика_за_Всеки', 'Кенгуру', 'Математика_за_Всеки', 'Кенгуру', 'Кенгуру']
for i in range(len(varArr1)):
    var1 = varArr1[i]
    var2 = varArr2[i]
    df2 = df[df[var1].notna() & df[var2].notna()]
    for vw in vv:
        var3 = vw+"_"+shArr1[i]
        var4 = vw+"_"+shArr2[i]
        df3 = df2[df2['top1adm'] == 0]
        df4 = df2[df2['top1adm'] == 1]
        df6 = df2[df2['top1adm'] == 2]
        scatter_double_color_plot(df3[var3].to_list(), df3[var4].to_list(), df4[var3].to_list(), df4[var4].to_list(), df6[var3].to_list(), df6[var4].to_list(), discArr1[i], discArr2[i], vw, 'sof_vs_all', 'неприети', 'приети_провинция', 'приети_София', 'by_acceptance')
        df7 = df2[(df2['top1adm'] == 0) & (df2['top1'] == 1)]
        scatter_triple_color_plot(df3[var3].to_list(), df3[var4].to_list(), df4[var3].to_list(), df4[var4].to_list(), df6[var3].to_list(), df6[var4].to_list(), df7[var3].to_list(), df7[var4].to_list(), discArr1[i], discArr2[i], vw, 'sof_vs_all', 'неприети_провинция', 'приети_провинция', 'приети_София', 'неприети_София', 'by_acceptance', '_miss')
        df3 = df2[df2['top4adm'] == 0]
        df4 = df2[df2['top4adm'] == 1]
        df6 = df2[df2['top4adm'] == 2]
        scatter_double_color_plot(df3[var3].to_list(), df3[var4].to_list(), df4[var3].to_list(), df4[var4].to_list(), df6[var3].to_list(), df6[var4].to_list(), discArr1[i], discArr2[i], vw, 'big4_vs_all', 'неприети', 'приети_провинция', 'приети_Сф_Ва_Бс_Пл', 'by_acceptance')
        df7 = df2[(df2['top4adm'] == 0) & (df2['top4'] == 1)]
        scatter_triple_color_plot(df3[var3].to_list(), df3[var4].to_list(), df4[var3].to_list(), df4[var4].to_list(), df6[var3].to_list(), df6[var4].to_list(), df7[var3].to_list(), df7[var4].to_list(), discArr1[i], discArr2[i], vw, 'big4_vs_all', 'неприети_провинция', 'приети_провинция', 'приети_Сф_Ва_Бс_Пл', 'неприети_Сф_Ва_Бс_Пл', 'by_acceptance', '_miss')

varArr1 = ['Резултат_Олимпиада', 'Резултат_Олимпиада', 'Резултат_Олимпиада', 'Резултат_ПМС', 'Резултат_ПМС']
varArr2 = ['Резултат_ПМС', 'Резултат_ПМС', 'Резултат_МатВс', 'Резултат_МатВс', 'Резултат_МатВс']
varArr3 = ['Резултат_МатВс', 'Резултат_Кенг', 'Резултат_Кенг', 'Резултат_Кенг', 'trans_keng']
shArr1 = ['olymp', 'olymp', 'olymp', 'pms', 'pms']
shArr2 = ['pms', 'pms', 'matvs', 'matvs', 'matvs']
shArr3 = ['matvs', 'keng', 'keng', 'keng', 'trans_keng']
discArr1 = ['Олимпиада', 'Олимпиада', 'Олимпиада', 'ПМС', 'ПМС']
discArr2 = ['ПМС', 'ПМС', 'Математика_за_Всеки', 'Математика_за_Всеки', 'Математика_за_Всеки']
discArr3 = ['Математика_за_Всеки', 'Кенгуру', 'Кенгуру', 'Кенгуру', 'Трансформирано_Кенгуру']

for i in range(len(varArr1)):
    for vw in vv:
        var1 = varArr1[i]
        var2 = varArr2[i]
        var3 = varArr3[i]
        shar1 = vw+"_"+shArr1[i]
        shar2 = vw+"_"+shArr2[i]
        shar3 = vw+"_"+shArr3[i]
        if var3 == 'trans_keng':
            shar3 = 'trans_keng'
        df2 = df[df[var1].notna() & df[var2].notna() & df[var3].notna()]
        scatter_plot_3d(df2[shar1].to_list(), df2[shar2].to_list(), df2[shar3].to_list(), discArr1[i], discArr2[i], discArr3[i], vw)
        df3 = df2[df2['top1adm'] == 0]
        df4 = df2[df2['top1adm'] == 1]
        df6 = df2[df2['top1adm'] == 2]
        scatter_double_color_plot_3d(df3[shar1].to_list(), df3[shar2].to_list(), df3[shar3].to_list(), df4[shar1].to_list(), df4[shar2].to_list(), df4[shar3].to_list(), df6[shar1].to_list(), df6[shar2].to_list(), df6[shar3].to_list(), discArr1[i], discArr2[i], discArr3[i], vw, 'sof_vs_all', 'неприети', 'приети_провинция', 'приети_София', 'joint_3d')
        df7 = df2[(df2['top1adm'] == 0) & (df2['top1'] == 1)]
        scatter_triple_color_plot_3d(df3[shar1].to_list(), df3[shar2].to_list(), df3[shar3].to_list(), df4[shar1].to_list(), df4[shar2].to_list(), df4[shar3].to_list(), df6[shar1].to_list(), df6[shar2].to_list(), df6[shar3].to_list(), df7[shar1].to_list(), df7[shar2].to_list(), df7[shar3].to_list(), discArr1[i], discArr2[i], discArr3[i], vw, 'sof_vs_all', 'неприети_провинция', 'приети_провинция', 'приети_София', 'неприети_София', 'joint_3d', '_miss')
        df3 = df2[df2['top4adm'] == 0]
        df4 = df2[df2['top4adm'] == 1]
        df6 = df2[df2['top4adm'] == 2]
        scatter_double_color_plot_3d(df3[shar1].to_list(), df3[shar2].to_list(), df3[shar3].to_list(), df4[shar1].to_list(), df4[shar2].to_list(), df4[shar3].to_list(), df6[shar1].to_list(), df6[shar2].to_list(), df6[shar3].to_list(), discArr1[i], discArr2[i], discArr3[i], vw, 'big4_vs_all', 'неприети', 'приети_провинция', 'приети_Сф_Ва_Бс_Пл', 'joint_3d')
        df7 = df2[(df2['top4adm'] == 0) & (df2['top4'] == 1)]
        scatter_triple_color_plot_3d(df3[shar1].to_list(), df3[shar2].to_list(), df3[shar3].to_list(), df4[shar1].to_list(), df4[shar2].to_list(), df4[shar3].to_list(), df6[shar1].to_list(), df6[shar2].to_list(), df6[shar3].to_list(), df7[shar1].to_list(), df7[shar2].to_list(), df7[shar3].to_list(), discArr1[i], discArr2[i], discArr3[i], vw, 'big4_vs_all', 'неприети_провинция', 'приети_провинция', 'приети_Сф_Ва_Бс_Пл', 'неприети_Сф_Ва_Бс_Пл', 'joint_3d', '_miss')

varArr1 = ['Резултат_Олимпиада', 'Резултат_Олимпиада', 'Резултат_Олимпиада', 'Резултат_ПМС', 'Резултат_ПМС', 'Резултат_МатВс']
varArr2 = ['Резултат_ПМС', 'Резултат_МатВс', 'Резултат_Кенг', 'Резултат_МатВс', 'Резултат_Кенг', 'Резултат_Кенг']
shArr1 = ['olymp', 'olymp', 'olymp', 'pms', 'pms', 'matvs']
shArr2 = ['pms', 'matvs', 'keng', 'matvs', 'keng', 'keng']
discArr1 = ['Олимпиада', 'Олимпиада', 'Олимпиада', 'ПМС', 'ПМС', 'Математика_за_Всеки']
discArr2 = ['ПМС', 'Математика_за_Всеки', 'Кенгуру', 'Математика_за_Всеки', 'Кенгуру', 'Кенгуру']
vv = ['aperc']

for i in range(len(varArr1)):
    for vw in vv:
        var1 = varArr1[i]
        var2 = varArr2[i]
        var3 = vw+"_"+shArr1[i]
        var4 = vw+"_"+shArr2[i]
        df2 = df[df[var1].notna() & df[var2].notna()]
        df5 = df2.groupby([var3,var4])[var3].agg('count').to_frame('count').reset_index()
        for k in range(len(mun)):
            mn = mun[k]
            mn_code = mun_code[k]
            df3 = df2[df2['Област'] != mn]
            df4 = df2[df2['Област'] == mn]
            df6 = df4[df4[mn_code] == 1]
            scatter_double_color_plot(df3[var3].to_list(), df3[var4].to_list(), df4[var3].to_list(), df4[var4].to_list(), df6[var3].to_list(), df6[var4].to_list(), discArr1[i], discArr2[i], vw, mn, 'извън_'+mn, 'неприети_'+mn, 'приети_'+mn, 'by_region', 'lightsteelblue')
