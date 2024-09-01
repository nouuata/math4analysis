import pandas as pd
import os

regions = ['BL','BR','DB','GB','HS','KR','KS','LV','MN','PL','PR','PV','PZ','RS','RZ','SF','SH','SL','SM','SO','SS','SZ','TR','VC','VD','VR','VT','YM']
path = 'data/'

comp = 'merged/'
if not os.path.exists(path + comp):
    os.makedirs(path + comp)

# KR   Х   pms   Х   keng
# KS   Х   pms matvs keng
# SZ olymp pms   Х   keng
# TR olymp  X    X   keng

how_type = ['inner', 'outer']

for region in regions:
    if region == 'KR':
        comp = 'pms'
        df_p = pd.read_csv(path + comp + '/' +comp+ '_' +region+ '_results.csv')
        print(df_p) 

        comp = 'keng'
        df_k = pd.read_csv(path + comp + '/' +comp+ '_' +region+ '_results.csv')
        print(df_k) 

        for h_type in how_type:
            df = pd.merge(df_p, df_k, on=['Име','Училище','Град','Област'], how=h_type)
            df['Резултат_Олимпиада'] = ""
            df['Резултат_МатВс'] = ""
            df = df[['Име','Училище','Град','Област','Резултат_Олимпиада','Резултат_ПМС','Резултат_МатВс','Резултат_Кенг']]
            if h_type == 'inner':
                df = df[0:0]
            comp = 'merged'
            df.to_csv(path + comp + '/' +comp+ '_' +region+ '_' + h_type + '.csv', index=False, quotechar="'")
    elif region == 'TR':
        comp = 'olymp'
        df_o = pd.read_csv(path + comp + '/' +comp+ '_' +region+ '_results.csv')
        print(df_o) 

        comp = 'keng'
        df_k = pd.read_csv(path + comp + '/' +comp+ '_' +region+ '_results.csv')
        print(df_k) 

        for h_type in how_type:
            df = pd.merge(df_o, df_k, on=['Име','Област'], how=h_type)
            df.rename(columns={'Училище_x': 'Училище', 'Град_x': 'Град'}, inplace=True)
            df.drop(['Училище_y', 'Град_y'], axis=1, inplace=True)
            df['Резултат_ПМС'] = ""
            df['Резултат_МатВс'] = ""
            df = df[['Име','Училище','Град','Област','Резултат_Олимпиада','Резултат_ПМС','Резултат_МатВс','Резултат_Кенг']]
            if h_type == 'inner':
                df = df[0:0]
            comp = 'merged'
            df.to_csv(path + comp + '/' +comp+ '_' +region+ '_' + h_type + '.csv', index=False, quotechar="'")
    elif region == 'KS':
        comp = 'pms'
        df_p = pd.read_csv(path + comp + '/' +comp+ '_' +region+ '_results.csv')
        print(df_p) 

        comp = 'matvs'
        df_m = pd.read_csv(path + comp + '/' +comp+ '_' +region+ '_results.csv')
        print(df_m) 

        comp = 'keng'
        df_k = pd.read_csv(path + comp + '/' +comp+ '_' +region+ '_results.csv')
        print(df_k) 

        for h_type in how_type:
            df = pd.merge(df_p, df_k, on=['Име','Училище','Град','Област'], how=h_type)
            dff = pd.merge(df, df_m, on=['Име','Област'], how=h_type)
            dff.rename(columns={'Училище_x': 'Училище', 'Град_x': 'Град'}, inplace=True)
            dff.drop(['Училище_y', 'Град_y'], axis=1, inplace=True)
            dff['Резултат_Олимпиада'] = ""
            dff = dff[['Име','Училище','Град','Област','Резултат_Олимпиада','Резултат_ПМС','Резултат_МатВс','Резултат_Кенг']]
            if h_type == 'inner':
                dff = dff[0:0]
            comp = 'merged'
            dff.to_csv(path + comp + '/' +comp+ '_' +region+ '_' + h_type + '.csv', index=False, quotechar="'")
    elif region == 'SZ':
        comp = 'olymp'
        df_o = pd.read_csv(path + comp + '/' +comp+ '_' +region+ '_results.csv')
        print(df_o) 

        comp = 'pms'
        df_p = pd.read_csv(path + comp + '/' +comp+ '_' +region+ '_results.csv')
        print(df_p) 

        comp = 'keng'
        df_k = pd.read_csv(path + comp + '/' +comp+ '_' +region+ '_results.csv')
        print(df_k) 

        for h_type in how_type:
            df = pd.merge(df_o, df_p, on=['Име','Училище','Град','Област'], how=h_type)
            dff = pd.merge(df, df_k, on=['Име','Училище','Град','Област'], how=h_type)
            dff['Резултат_МатВс'] = ""
            dff = dff[['Име','Училище','Град','Област','Резултат_Олимпиада','Резултат_ПМС','Резултат_МатВс','Резултат_Кенг']]
            if h_type == 'inner':
                dff = dff[0:0]
            comp = 'merged'
            dff.to_csv(path + comp + '/' +comp+ '_' +region+ '_' + h_type + '.csv', index=False, quotechar="'")
    else:
        comp = 'olymp'
        df_o = pd.read_csv(path + comp + '/' +comp+ '_' +region+ '_results.csv')
        print(df_o) 
        print(df_o.columns) 

        comp = 'pms'
        df_p = pd.read_csv(path + comp + '/' +comp+ '_' +region+ '_results.csv')
        print(df_p) 

        comp = 'matvs'
        df_m = pd.read_csv(path + comp + '/' +comp+ '_' +region+ '_results.csv')
        print(df_m) 

        comp = 'keng'
        df_k = pd.read_csv(path + comp + '/' +comp+ '_' +region+ '_results.csv')
        print(df_k) 

        # df_e = pd.read_csv('evrika_SF_results.csv')
        # print(df_e) 

        if region == 'BL' or region == 'SF':
            for h_type in how_type:
                df = pd.merge(df_o, df_p, on=['Име','Училище','Град','Област'], how=h_type)
                dff = pd.merge(df, df_k, on=['Име','Училище','Град','Област'], how=h_type)
                dff = pd.merge(dff, df_m, on=['Име','Област'], how=h_type)
                dff.rename(columns={'Училище_x': 'Училище', 'Град_x': 'Град'}, inplace=True)
                dff.drop(['Училище_y', 'Град_y'], axis=1, inplace=True)
                dff = dff[['Име','Училище','Град','Област','Резултат_Олимпиада','Резултат_ПМС','Резултат_МатВс','Резултат_Кенг']]
                comp = 'merged'
                dff.to_csv(path + comp + '/' +comp+ '_' +region+ '_' + h_type + '.csv', index=False, quotechar="'")
        elif region == 'MN':
            for h_type in how_type:
                df = pd.merge(df_m, df_p, on=['Име','Училище','Град','Област'], how=h_type)
                dff = pd.merge(df, df_k, on=['Име','Училище','Град','Област'], how=h_type)
                dff = pd.merge(dff, df_o, on=['Име','Област'], how=h_type)
                dff.rename(columns={'Училище_x': 'Училище', 'Град_x': 'Град'}, inplace=True)
                dff.drop(['Училище_y', 'Град_y'], axis=1, inplace=True)
                dff = dff[['Име','Училище','Град','Област','Резултат_Олимпиада','Резултат_ПМС','Резултат_МатВс','Резултат_Кенг']]
                comp = 'merged'
                dff.to_csv(path + comp + '/' +comp+ '_' +region+ '_' + h_type + '.csv', index=False, quotechar="'")
        elif region == 'GB':
            for h_type in how_type:
                df = pd.merge(df_o, df_k, on=['Име','Училище','Град','Област'], how=h_type)
                dff = pd.merge(df, df_p, on=['Име','Област'], how=h_type)
                dff.rename(columns={'Училище_x': 'Училище', 'Град_x': 'Град'}, inplace=True)
                dff.drop(['Училище_y', 'Град_y'], axis=1, inplace=True)
                dff = pd.merge(dff, df_m, on=['Име','Област'], how=h_type)
                dff.rename(columns={'Училище_x': 'Училище', 'Град_x': 'Град'}, inplace=True)
                dff.drop(['Училище_y', 'Град_y'], axis=1, inplace=True)
                dff = dff[['Име','Училище','Град','Област','Резултат_Олимпиада','Резултат_ПМС','Резултат_МатВс','Резултат_Кенг']]
                comp = 'merged'
                dff.to_csv(path + comp + '/' +comp+ '_' +region+ '_' + h_type + '.csv', index=False, quotechar="'")
        elif region == 'YM':
            for h_type in how_type:
                df = pd.merge(df_o, df_p, on=['Име','Област'], how=h_type)
                dff = pd.merge(df, df_m, on=['Име','Област'], how=h_type)
                dff.drop(['Училище_x', 'Град_x', 'Училище_y', 'Град_y'], axis=1, inplace=True)
                dff = dff[['Име','Училище','Град','Област','Резултат_Олимпиада','Резултат_ПМС','Резултат_МатВс']]
                dff = pd.concat([dff, df_k], axis=0)
                if h_type == 'inner':
                    dff = dff[0:0]
                comp = 'merged'
                dff.to_csv(path + comp + '/' +comp+ '_' +region+ '_' + h_type + '.csv', index=False, quotechar="'")
        elif region == 'SO':
            for h_type in how_type:
                df = pd.merge(df_o, df_p, on=['Име','Училище','Град','Област'], how=h_type)
                dff = pd.merge(df, df_m, on=['Име','Училище','Град','Област'], how=h_type)
                dff = pd.merge(dff, df_k, on=['Име','Област'], how=h_type)
                dff.rename(columns={'Училище_x': 'Училище', 'Град_x': 'Град'}, inplace=True)
                dff.drop(['Училище_y', 'Град_y'], axis=1, inplace=True)
                dff = dff[['Име','Училище','Град','Област','Резултат_Олимпиада','Резултат_ПМС','Резултат_МатВс','Резултат_Кенг']]
                comp = 'merged'
                dff.to_csv(path + comp + '/' +comp+ '_' +region+ '_' + h_type + '.csv', index=False, quotechar="'")
        # if region == BR DB HS LV PL PR PV PZ RS RZ SH SL SM SS VC VD VR VT
        else:
            for h_type in how_type:
                df = pd.merge(df_o, df_p, on=['Име','Училище','Град','Област'], how=h_type)
                dff = pd.merge(df, df_m, on=['Име','Училище','Град','Област'], how=h_type)
                dff = pd.merge(dff, df_k, on=['Име','Училище','Град','Област'], how=h_type)
                comp = 'merged'
                dff.to_csv(path + comp + '/' +comp+ '_' +region+ '_' + h_type + '.csv', index=False, quotechar="'")

how_type = ['results','results','results','results','inner', 'outer']
comps = ['olymp','pms','matvs','keng','merged','merged']

for idx in range(len(how_type)):
    h_type = how_type[idx]
    comp = comps[idx]
    d_frames = []
    for region in regions:
        d_frames.append(pd.read_csv(path + comp + '/' +comp+ '_' +region+ '_' +h_type+ '.csv'))
        dff = pd.concat(d_frames)
        dff.to_csv(path +comp+ '_ALL_' + h_type + '.csv', index=False, quotechar="'")