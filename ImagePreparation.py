
import os
import shutil
from skimage.transform import resize
from matplotlib import image as mpimg
import matplotlib.pyplot as plt
from multiprocessing import Pool

class ImagePreparation:
    def __init__(self, directory,source, replace=True, replace_characters="/\\|?*\":'(),.;“”„"):
        self.h_l, self.w_l = [], []
        self.directory = directory
        self.replace = replace
        self.source = source
        self.replace_characters = replace_characters
        self.translation_table = str.maketrans("","",replace_characters)
        self.num_of_authors = 8
        self.delete_array = ['ę', 'ł', 'x', 'h', 'w', 'f', 'j', 'd', 'ń', 'i','b', 'ż', 'k', 'e', 'o', 'n', 'c', '+', 'ó',
                 't', 'u', 'l', 'ź', 'q', 'a', 'r', 'm', 'ą', '!', 'y', 's', 'p', 'ć', 'z', 'g', 'v', 'ś']
        
    def _crop(self,n, max_n):
        return max(0,n) if n <= max_n else max_n
    
    def _trim_image(self,image, trim_percentage=0.9):
        new_height = int(image.shape[0] * trim_percentage)
        new_width = int(image.shape[1] * trim_percentage)
        
        offset_height = (image.shape[0] - new_height) // 2
        offset_width = (image.shape[1] - new_width) // 2
        
        cropped_image = image[offset_height:offset_height+new_height, offset_width:offset_width+new_width]
        
        return cropped_image

    def _merge(self,l, start, end):
        sub_merged = ""
        for o in l[start:end+1]:
            sub_merged = sub_merged+o
        merged = l[:start] + [sub_merged] + l[end+1:]
        return merged


    def _save_words_to_files(self,author_no):
        global_counter = 0
        file_desc_name = self.source + str(author_no + 1) + "/word_places.txt"
        file_desc_ptr = open(file_desc_name, 'r', encoding='windows-1250')
        text = file_desc_ptr.read()
        lines = text.split('\n')
        number_of_lines = lines.__len__() - 1
        row_values = lines[0].split()
        number_of_values = row_values.__len__()

        num_of_words = 0
        image_file_name_prev = ""
        subimage_dir = self.directory+'/'
        
        if not os.path.exists(subimage_dir):
            os.makedirs(subimage_dir)   
            
        for i in range(number_of_lines):
            row_values = lines[i].split()
            
            if len(row_values) > 6:
                row_values = self._merge(row_values,1,len(row_values)-5)
            elif len(row_values) < 6:
                continue

            if row_values[0] != '%':
                num_of_words += 1
                number_of_values = len(row_values)
                
                image_file_name = self.source + str(author_no + 1) + "/" + row_values[0][1:-1]

                if image_file_name != image_file_name_prev:   
                    image = mpimg.imread(str(image_file_name))
                    image_file_name_prev = image_file_name
                word = row_values[1]

                if self.replace:
                    word = word.translate(self.translation_table)

                if word == "<brak>":
                    continue
                
                row1, column1, row2, column2 = int(row_values[2]), int(row_values[3]), \
                    int(row_values[4]), int(row_values[5])

                height, width = len(image), len(image[0])
                row1, row2 =  self._crop(row1,height), self._crop(row2,height)
                column1, column2 =  self._crop(column1,width), self._crop(column2,width)

                subimage = image[min(row1,row2):max(row1,row2),
                                min(column1,column2):max(column1,column2)] 
                
                self.h_l.append(len(subimage))
                self.w_l.append(len(subimage[0]))
                
               
                mpimg.imsave(subimage_dir+word +"_"+str(author_no)+"_"+str(global_counter)+".png",subimage)
                global_counter += 1
            


        file_desc_ptr.close()

    def _sort_files(self,folder_path):
        
        files = os.listdir(folder_path)
        
        file_dict = {}
        for file_name in files:
            file_path = folder_path +'/'+ file_name
            if os.path.isfile(file_path):
                base_name = os.path.splitext(file_name)[0]
                base_name =  base_name.split("_",1) 
                base_name = base_name[0].lower()
                file_dict.setdefault(base_name, []).append(file_path)

        for base_name, file_paths in file_dict.items():
            new_folder_path = folder_path +"/"+ base_name
            os.makedirs(new_folder_path, exist_ok=True)
            
            for file_path in file_paths:
                file_name = os.path.basename(file_path)
                new_file_path = new_folder_path + "/"+ file_name
                shutil.move(file_path, new_file_path)

    def _delete_folders_with_additional_characters(self,directory, delete_array):
        folder_list = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
        for folder in folder_list:
            for char in folder:
                if char not in delete_array:
                    folder_path = os.path.join(directory, folder)
                    print(f"Deleting folder: {folder_path} is not in:{char}")
                    shutil.rmtree(folder_path)
                    break





    def prepare_images(self):
        
        if os.path.exists(self.directory):
            shutil.rmtree(self.directory)
            os.mkdir(self.directory)
        # pool = Pool(processes=8)
        for i in range(self.num_of_authors):
            self._save_words_to_files(i)
            # pool.apply_async(self._save_words_to_files, args=(i,))
            print("processing author ", i)
        # pool.close()
        # pool.join()
        self._sort_files(self.directory)
        print('sorting files')
        self._delete_folders_with_additional_characters(self.directory, self.delete_array)
