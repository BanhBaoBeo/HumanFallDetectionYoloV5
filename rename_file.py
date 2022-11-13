import os
import glob
import re

#Tạo hàm đổi tên
def rename(folder_path, extention):
    #Biến đếm để lưu tên của bức ảnh: 1.png, 2.png,...., 1000.png
    count = 1
    #Truy vấn folder/file ở trong folder_path truyền vào
    for folder_name in os.listdir(folder_path):
        # Xây dựng đường dẫn để dẫn tới tệp cần đổi tên 
        #Đường dẫn này sẽ đi qua sub_folder đến tệp ảnh cần đổi tên
        subfolder_path = os.path.join(folder_path, folder_name)
        print(subfolder_path)

        #Tạo mảng chứa tất cả các file.png có trong sub_folder mình vừa truy vấn
        all_files = glob.glob(subfolder_path + r'\*\*' + extention)
        #In thử ra xem kết quả
        print(all_files)
        #Nếu mảng rỗng tức trong folder ban đầu không có subfolder nào
        #Folder ban đầu chỉ chứa ảnh cần đổi tên
        if len(all_files) == 0: 
            file_location = re.findall('(.+)\\\.+' + extention, subfolder_path)
            print(file_location)           
            destination = str(count)
            os.rename(subfolder_path, file_location[0] + "\\" + str(0) * (4 - len(destination)) + destination + extention)
            count += 1
        #Else nếu folder ban đầu chứa ít nhất 1 subfolder
        else:
            #Duyệt các file.png trong các subfolder
            for file in all_files:
                destination = str(count)
                file_location = re.findall('(.+)\\\.+' + extention, file)
                print(file_location)
                #Hàm đổi tên với quy tắc tên mới sẽ thêm các số 0 bên trên cho đủ 4 chữ số
                #Ví dụ 0001.png
                os.rename(file, file_location[0] + "\\" + str(0) * (4 - len(destination)) + destination + extention)
                count += 1
    print('All Files Renamed')

#Truyền đường dẫn cần đổi tên vào
#rename(folder_path)

def findNotExist(folder_path_1, folder_path_2):
    file_list_1 = list()
    file_list_2 = list()
    notExist_image = list()
    notExist_label = list()
    final_list = list()
    for file_name_1 in os.listdir(folder_path_1):
        file_list_1.append(file_name_1.split(".")[0])
    for file_name_2 in os.listdir(folder_path_2):
        file_list_2.append(file_name_2.split(".")[0])
    for t in file_list_1: 
        if t not in file_list_2: 
            notExist_label.append(t)
    for t in file_list_2: 
        if t not in file_list_1: 
            notExist_image.append(t)    
    return [notExist_label, notExist_image]

#print(findNotExist(r"C:\Users\LENOVO\Desktop\Code\AIP_project\rgb\rgb", r"C:\Users\LENOVO\Desktop\Code\AIP_project\labels_my-project-name_2022-10-24-04-06-06"))
rename(r"data\images\train", ".png")
rename(r"data\labels\train", ".txt")