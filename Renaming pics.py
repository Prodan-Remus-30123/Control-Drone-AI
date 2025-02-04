import os
def rename_imgs(path,start_index):
    files=os.listdir(path)
    # Filter out only image files (you can adjust the condition if needed)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    # Sort the image files to ensure they are processed in order
    image_files.sort()

    for i,filename in enumerate (image_files,start=start_index):
        new_filename= str(i)+ os.path.splitext(filename)[1]
        #Rename the file
        os.rename(os.path.join(path,filename), os.path.join(path,new_filename))


def openfolders(parent_folder):
    for folder_name in os.listdir(parent_folder):
        folder_path=os.path.join(parent_folder, folder_name)

        if(os.path.isdir(folder_path)):
            if folder_name == "Down":
                start_index = 1847
            elif folder_name == "Left":
                start_index = 1825
            elif folder_name == "Right":
                start_index = 2239
            elif folder_name == "Rock":
                start_index = 2040
            elif folder_name == "Stay":
                start_index = 1915
            elif folder_name == "Up":
                start_index = 1863
            rename_imgs(folder_path,start_index)

parent_folder="E:/Facultate/Licenta/Incercari Curs AI/Data3/test"
openfolders(parent_folder)