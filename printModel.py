import pickle
import xlwt
# 创建一个workbook 设置编码
if __name__ == '__main__':

    with open("C:\\Users\\lenovo\\Desktop\\KCGN_Yelp_825401_reg_0.05_batch_2048_lr_0.01_decay_0.98_hide_32_Layer_[32]_slope_0.4_top_10_fuse_mean_timeStep_360_lam_[0.1, 0.001]sigmoid.his", 'rb')as fs:
        data=pickle.load(fs)
        data_HR=data['HR']
        data_NDCG=data['NDCG']
        workbook = xlwt.Workbook(encoding='utf-8')
        # 创建一个worksheet
        worksheet = workbook.add_sheet('My Worksheet')

        # 写入excel
        # 参数对应 行, 列, 值
        for i in range(len(data_HR)):
            worksheet.write(i, 0, data_HR[i])
            worksheet.write(i, 1, data_NDCG[i])

        # 保存
        workbook.save('Excel_test_o.xls')
        print(data)