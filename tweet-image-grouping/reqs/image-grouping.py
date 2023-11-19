#!/usr/bin/env python
# coding: utf-8

# ## Compute Similarities between images
# Using Structural Similarity Index
# 
# Importing all necessary package

# In[1]:


# import the necessary packages
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils
import urllib

import pymysql.cursors
import pandas as pd
import numpy as np
import uuid
import time
import os
import string
import re
import image_similarity_measures
import urllib.request as rq

# create matrix
import numpy as np
import array as arr
from tqdm import tqdm
from PIL import Image
import requests

from PIL import Image
from image_tools.sizes import resize_and_crop
from image_similarity_measures.quality_metrics import ssim, sam, sre


print('done importing modules')


# ### Querying Data from server
# based on given job id
# 
# ### Input as proces started
# Record header and parameter information
# 
#  * Kesepakatan status di kolom screen_analisis_ai.status
#  * 1 --> baru diinput
#  * 2 --> lagi dikerjakan
#  * 3 --> proses berhasil
#  * 4 --> proses gagal
#  *
#  
#  * Kesepakatan jenis analisa AI
#  * 1 --> Analisa Cluster
#  * 2 --> Analisa image clustering
#  * 3 --> Analisa sentiment

# In[2]:


from datetime import datetime

# Connect to the database
connection = pymysql.connect(host='202.157.176.225',
                             user='cekmedsos_db',
                             password='kuku838485*#',
                             database='cekmedsos_database',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

cursor = connection.cursor()

# get available jobs from database server, first come first serve
sql = "select id, hastag, `parameter` from screen_analisis_ai where active = 1 and status = 1 and jenis_analisa = 2 order by created asc, id asc limit 1"

print(sql)

row_count = cursor.execute(sql)

if(row_count == 0):
    # get out, nothing to do
    print('Zero jobs, quitting now')
    quit()

result = cursor.fetchall()
database_keyword_id = result[0]['hastag']
similarity_treshold = result[0]['parameter']
i_process_id = result[0]['id']
screen_name = ''

print(database_keyword_id)
print(similarity_treshold)
print(i_process_id)


# ### Mark Process as Running

# In[3]:


# Connect to the database
connection = pymysql.connect(host='202.157.176.225',
                             user='cekmedsos_db',
                             password='kuku838485*#',
                             database='cekmedsos_database',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
cursor = connection.cursor()

# Prepare SQL Statement
print(i_process_id)
sql = "update screen_analisis_ai set status = 2, last_status_update = now(), start_process = now() where id = %s"
# execute
cursor.execute(sql, i_process_id)

#
# Create Header Record
sql = "insert into ret_analysis_header (job_id, datetime_start, user_id) values (%s, %s, %s)"
# Execute the query
cursor.execute(sql, (str(i_process_id), datetime.now(), 1 ))

#
# Create Parameter Record
sql = "insert into ret_analysis_parameter (job_id, param_id, param_name, param_value) values (%s, %s, %s, %s)"
# Execute the query
cursor.execute(sql, (i_process_id, 1, 'Similarity Treshold', similarity_treshold))
cursor.execute(sql, (i_process_id, 1, 'DB_ID', database_keyword_id))

# commit record
connection.commit()


# ### Query data from RDBMS

# In[4]:


i_db_id = database_keyword_id
## similarity_treshold = 0.8

#
# Query to get tweet data, apply analitics to this dataset
#

# Connect to the database
connection = pymysql.connect(host='202.157.176.225',
                             user='cekmedsos_db',
                             password='kuku838485*#',
                             database='cekmedsos_database',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

cursor = connection.cursor()
s_query_string = "select a.id,a.db_id,c.tweet_id,c.filename from ret_available_db a inner join ret_tweet b on a.db_id = b.db_id inner join media_files c on a.db_id = c.db_id and b.id = c.tweet_id where 	a.db_id = " + str(i_db_id)
    
print(s_query_string)
df = pd.read_sql(s_query_string, con=connection)

# Close Connection
connection.close()

# see result
# df


# ### Defining Functions
# To compare images

# In[5]:


# load image from url
def urlToImage(url):
    # download image,convert to a NumPy array,and read it into opencv
    resp = rq.urlopen(url)
    img = np.asarray(bytearray(resp.read()),dtype="uint8")
    img = cv2.imdecode(img,cv2.IMREAD_COLOR)

    #return the image
    return img

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)
    
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
    
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
	# show the images
	plt.show()
    
def largest_in_col(arr,nCol):
    # 
    # Find largest value of col nCol on 2D arr
    #
    
    # init value
    max_val = arr[0][nCol]
    # also, remember index
    row_index = 0
    
    for x in range(0, len(arr)):
        if arr[x][nCol] > max_val:
            max_val = arr[x][nCol]
            row_index = x
        
    return max_val,row_index

def largest_in_row(arr,nRow):
    # 
    # Find largest value of row nRow on 2D arr
    #
    
    # initial value
    max_val = arr[nRow][0]
    # also, remember index
    col_index = 0
    
    for x in range(0, len(arr)):
        if arr[nRow][x] > max_val:
            max_val = arr[nRow][x]
            col_index = x
            
    return max_val,col_index  

    

print('done implement function')


# In[6]:


# imgUrl = "https://cekmedsos.com/uploads/twimg/" + df['filename'][1]
# img_base = Image.open(requests.get(imgUrl, stream=True).raw)
# wa, ha = img_base.size
# print(wa,ha)

# imgUrl = "https://cekmedsos.com/uploads/twimg/" + df['filename'][2]
# img_compare = Image.open(requests.get(imgUrl, stream=True).raw)

# fig = plt.figure('test')
# ax = fig.add_subplot(1, 2, 1)
# plt.imshow(img_base, cmap = plt.cm.gray)
# plt.axis("off")

# plt.show()

# size_check = check_image_size(img_base, img_compare)
# print('size_check = ' + str(size_check))




# img_base = cv2.cvtColor(np.float32(img_base), cv2.COLOR_BGR2GRAY)
# img_compare = cv2.cvtColor(np.float32(img_compare), cv2.COLOR_BGR2GRAY)

# compare_images(img_base, img_compare, 'test')


# ### Perform Image Compare
# By loading the images
# 
# 1. Compares itself
# 2. Compare to Contrast Editing
# 3. Compare to PS editing (Black box)
# 

# In[ ]:


# len(df)


# In[7]:


import tempfile

def match_size(imgA, imgB):
    tf = tempfile.NamedTemporaryFile(suffix='.jpg')
    tf_ = tempfile.NamedTemporaryFile(suffix='.jpg')
    
    # which one have more pixels?
    pxA = imgA.shape[0]*imgA.shape[1]
    pxB = imgB.shape[0]*imgB.shape[1]
    
    # handle odd pixel size
    #print('W/H= ' + str(imgA.shape[0]) + '/' + str(imgA.shape[1]))
    #print('W/H= ' + str(imgB.shape[0]) + '/' + str(imgB.shape[1]))

    if(pxA < pxB):
        # print('resize imgB to match imgA')
        # save temp file imgB
        cv2.imwrite(tf.name,imgB)
        imgRes = resize_and_crop(
                        tf.name, 
                        (imgA.shape[1],imgA.shape[0]), #set width and height to match img1
                        crop_origin="middle"
                        )
        # save temp file imgA        
        cv2.imwrite(tf_.name,imgA)
        imgRes_ = resize_and_crop(
                        tf_.name, 
                        (imgA.shape[1],imgA.shape[0]), #set width and height to match img1
                        crop_origin="middle"
                        )        
        
        del tf
        del tf_
    else:
        # print('resize imgA to match imgB')
        # save temp file imgB
        cv2.imwrite(tf.name,imgA)
        imgRes = resize_and_crop(
                        tf.name, 
                        (imgB.shape[1],imgB.shape[0]), #set width and height to match img1
                        crop_origin="middle"
                        )
        # save temp file imgB       
        cv2.imwrite(tf_.name,imgB)
        imgRes_ = resize_and_crop(
                        tf_.name, 
                        (imgB.shape[1],imgB.shape[0]), #set width and height to match img1
                        crop_origin="middle"
                        )             
        del tf
        del tf_
        
    # need to check for pixel difference between images
    if imgRes.size[0] != imgRes_.size[0]:
        # print('diff width')
        # which one bigger?
        if imgRes.size[0] > imgRes_.size[0]:
            # crop to imgRes to match imgRes_
            # print('cropping imgRes to match imgRes_')
            imgRes = imgRes.crop((0,0,imgRes_.size[0], imgRes_.size[1]))
        else:
            # crop to imgRes_ to match imgRes
            # print('cropping imgRes_ to match imgRes')
            imgRes_ = imgRes_.crop((0,0,imgRes.size[0], imgRes.size[1]))           
        
    if imgRes.size[1] != imgRes_.size[1]:
        # print('diff height')
        # which one bigger?
        if imgRes.size[1] > imgRes_.size[1]:
            # crop to imgRes to match imgRes_
            #print('cropping imgRes to match imgRes_')
            imgRes = imgRes.crop((0,0,imgRes_.size[0], imgRes_.size[1]))
        else:
            # crop to imgRes_ to match imgRes
            #print('cropping imgRes_ to match imgRes')
            imgRes_ = imgRes_.crop((0,0,imgRes.size[0], imgRes.size[1]))            
            
        
    return imgRes, imgRes_
#
# END Function
#


# In[ ]:


# imgA = urlToImage("https://cekmedsos.com/uploads/twimg/60b5246c4827f.jpg")
# imgB = urlToImage("https://cekmedsos.com/uploads/twimg/60b5246b3c5ad.jpg")

# img_one, img_two = match_size(imgB, imgA)
# img_one = cv2.cvtColor(np.float32(img_one), cv2.COLOR_BGR2GRAY)
# img_two = cv2.cvtColor(np.float32(img_two), cv2.COLOR_BGR2GRAY)

# #print(str(img_one.shape[0]) + '/' + str(img_one.shape[1]))
# #print(str(img_two.shape[0]) + '/' + str(img_two.shape[1]))

# #print(img_two.size)

# # plt.imshow(imgA)
# #.imshow(imgB)


# s = ssim(img_one, img_two)

# # image-similarity-measures --org_img_path test-3.jpg --pred_img_path test-4.jpg --metric=all
# print('Similarity Index')
# print(s)

# print('finished')


# In[8]:


st = similarity_treshold
cluster_no = 1
s_score = 0
s_score_current = 0
i_current_cluster = 0

def check_image_size(imageA, imageB):
    wa = imageA.shape[0]
    ha = imageA.shape[1]
    
    wb = imageB.shape[0]
    hb = imageB.shape[1]
    
    if(wa == wb and ha == hb):
        return True
    else:
        return False

#create zero element array
#col 0 => index base tweet
#col 1 => cluster number
#col 2 => similarity score
base_tweet = []

ssim_matrix = np.zeros(( len(df), len(df) ), dtype=np.dtype('f4'))

with tqdm(total=( (len(df)*len(df)/2))-(len(df)/2))  as pbar:
    ## Kolom
    for j in range(0, len(df)):
        # get url
        imgUrl = "https://cekmedsos.com/uploads/twimg/" + df['filename'][j]
        img_base = urlToImage(imgUrl)
        
        ## Baris
        for i in range(0,len(df)):
            if (j<i):
                # get url
                imgUrl = "https://cekmedsos.com/uploads/twimg/" + df['filename'][i]
                # print(imgUrl)
                img_compare = urlToImage(imgUrl)
                
                # do image check size
                if(check_image_size(img_base, img_compare)):
                    # do check images ssim
                    
                    img_compare_one = cv2.cvtColor(np.float32(img_base), cv2.COLOR_BGR2GRAY)
                    img_compare_two = cv2.cvtColor(np.float32(img_compare), cv2.COLOR_BGR2GRAY)
                    
                    s = ssim(img_compare_one, img_compare_two)
                    # s2 = sam(img_compare_one, img_compare_two)
                    # s3 = sre(img_compare_one, img_compare_two)

                    ssim_matrix[i,j] = s
                    # sam_matrix[i,j] = s2
                    # sre_matrix[i,j] = s3
                    
                    # releasing object
                    del img_compare_one
                    del img_compare_two
                    
                else:
                    # resize image to match
                    imgA, imgB = match_size(img_base, img_compare)
                    
                    #print('Size A: W:' + str(imgA.size[0]) + 'H: ' + str(imgA.size[1]))
                    #print('Size B: W:' + str(imgB.size[0]) + 'H: ' + str(imgB.size[1]))
                    
                    if(imgA.size[0]*imgA.size[1] == imgB.size[0]*imgB.size[1]):
                        img_compare_one = cv2.cvtColor(np.float32(imgA), cv2.COLOR_BGR2GRAY)
                        img_compare_two = cv2.cvtColor(np.float32(imgB), cv2.COLOR_BGR2GRAY)
                    
                        s = ssim(img_compare_one, img_compare_two)
                        # s2 = sam(img_compare_one, img_compare_two)
                        # s3 = sre(img_compare_one, img_compare_two)
                        
                        ssim_matrix[i,j] = s
                        # sam_matrix[i,j] = s2
                        # sre_matrix[i,j] = s3
                        
                        # releasing object
                        del img_compare_one
                        del img_compare_two
                        del imgA
                        del imgB
                    else:
                        ssim_matrix[i,j] = 0.0
                    
                pbar.update(1)
            # if (j<i):
        # for i in range(0,len(df)):
        # selesai satu baris
    # for j in range(0, len(df)):
    # selesai satu kolom
    # check for any cluster info at this column
    
#with tqdm(total=( (len(df)*len(df)/2))-(len(df)/2))  as pbar:
            
                 
pbar.close()
print(ssim_matrix)


# In[9]:


# from numpy import asarray
# from numpy import savetxt

# savetxt('data.csv', ssim_matrix, delimiter=',')


# ## SSIM Matrix Analysis
# Goals :
# - create base-tweet
# - create cluster result

# In[10]:


largest_s = 0.0

#create zero element array
#col 0 => index base tweet
#col 1 => cluster number
#col 2 => similarity score
base_tweet = []

#initial cluster number
i_current_cluster = 0

for j in range(0, len(ssim_matrix)):
    largest_s = largest_in_col(ssim_matrix,j)
    if (largest_s[0] > similarity_treshold):
        print('largest s-index in column ' + str(j) + ' is: '               + str(largest_s[0]) +               ' on rows# ' + str(largest_s[1]))
        
        # cari di baris yang kolomnya ini, untuk baris < kolom skrg
        largest_s_row = largest_in_row(ssim_matrix,j)
        print('largest s-index in row ' + str(j) + ' is: '               + str(largest_s_row[0]) +               ' on col# ' + str(largest_s_row[1]))
        
        if (largest_s_row[0] < j):
            if (largest_s_row[0] < similarity_treshold):
                # add new cluster index
                # print('tambah')
                print(similarity_treshold)
                i_current_cluster = i_current_cluster + 1
                base_tweet.append([j,i_current_cluster,largest_s[0]])
            
# print(base_tweet)
# append base 0
base_tweet.append([0, 0, 0])


# In[11]:


base_max = 0.0
base_now = 0.0
base_index = 0
cluster_no = 0
base_content = 0

#create zero element array
#col 0 => tweet id
#col 1 => cluster no
cluster_result = []

for j in range(0, len(ssim_matrix)):
    # compare to base tweet
    # print('baris: ' + str(j))
    
    # cari nilai paling tinggi dari 3 base tweet
    for val in base_tweet:
        # print('sim index kolom: '+ str(val[0]) + ' adalah: ' + str(ssim_matrix[j,val[0]]))
        base_now = ssim_matrix[j,val[0]]
        if base_now > base_max:
            base_max = base_now
            base_index = val[0]
            cluster_no = val[1]
    
    if(base_max > similarity_treshold):
        # print('max dari baris: ' + str(j) + ' adalah: ' + str(base_max) + ' pada index: ' + str(cluster_no))
        ## append to cluster-result
        cluster_result.append([df['tweet_id'][j], cluster_no])
    else:        
        # jika ada dalam base-tweet, masukin sebagai cluster
        base_content = 0
        for val in base_tweet:
            if(j == val[0]):
                # insert as cluster
                cluster_result.append([df['tweet_id'][j], val[1]])
                # markdown as base
                base_content = 1
                
        # non cluster
        if(base_content == 0):
            cluster_result.append([df['tweet_id'][j], 0])
    
    # reset base_max
    base_max = 0
    base_index = 0
    cluster_no = 0

# print(cluster_result)


# In[12]:


## save cluster result and base_tweet
from numpy import asarray
from numpy import savetxt

savetxt('cluster_result.csv', cluster_result, delimiter=',')
savetxt('base_tweet.csv', base_tweet, delimiter=',')

# for i in range(0, len(df)):
#     imgUrl = "https://cekmedsos.com/uploads/twimg/" + df['filename'][i]
#     print(imgUrl)
#     # loads this images
#     img_base = Image.open(requests.get(imgUrl, stream=True).raw)
#     # img_base = cv2.imread(requests.get(imgUrl, stream=True).raw)
#     img_base.save(str(i) + '.jpg',"PNG")
    
# # (score, diff) = compare_images(original, shopped, 'Original vs Shopped')

# (score,diff) = ssim(original, shopped, full=True)
# diff = (diff * 255).astype("uint8")
# print("SSIM: {}".format(score))


# ### Inserting to database
# 
# 1. Base Tweet
# 2. Cluster Result
# 3. Record Processing Time
# 4. Marking as finished Job

# ### 1. Base tweet insert

# In[13]:


# Connect to the database
connection = pymysql.connect(host='202.157.176.225',
                             user='cekmedsos_db',
                             password='kuku838485*#',
                             database='cekmedsos_database',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

cursor = connection.cursor()

#
# Create Base Tweet Record
sql = "insert into ret_base_tweet (job_id, tweet_id, cluster_id) values (%s, %s, %s)"

## inserting base taweet
for i in range(0,len(base_tweet)):
    # Execute the query
    if ( base_tweet[i][0] == 0):
         #print(sql,          (i_process_id, 0, base_tweet[i][1]))   
         cursor.execute(sql, (i_process_id, 0, base_tweet[i][1]))
    else:
         cursor.execute(sql, (i_process_id, df['tweet_id'][base_tweet[i][0]], base_tweet[i][1]))
         #print(sql,          (i_process_id, df['tweet_id'][base_tweet[i][0]], base_tweet[i][1]))
    
connection.commit()
connection.close()

print('finished inserting base tweet record')


# ### 2. Cluster Result

# In[14]:


# Connect to the database
connection = pymysql.connect(host='202.157.176.225',
                             user='cekmedsos_db',
                             password='kuku838485*#',
                             database='cekmedsos_database',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

cursor = connection.cursor()

#
# Create Base Tweet Record
sql = "insert into ret_cluster_result (job_id, tweet_id, cluster_no) values (%s, %s, %s)"

## inserting base taweet
for i in range(0,len(cluster_result)):
    # Execute the query
    cursor.execute(sql, (i_process_id, cluster_result[i][0], cluster_result[i][1]))
    # cluster_result[i][0]
    # print(sql,          (i_process_id, cluster_result[i][0], cluster_result[i][1]))
    
connection.commit()
connection.close()

print('finished inserting base tweet record')


# ### 3. Record Processing Time

# In[15]:


# Connect to the database
connection = pymysql.connect(host='202.157.176.225',
                             user='cekmedsos_db',
                             password='kuku838485*#',
                             database='cekmedsos_database',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

cursor = connection.cursor()

#
# Create Parameter Record
sql = "insert into ret_analysis_parameter (job_id, param_id, param_name, param_value) values (%s, %s, %s, %s)"
# Execute the query
cursor.execute(sql, (i_process_id, 1, '#Tweet Processed',len(df)))

#
# Create Tweet Cluster Record
sql = "update ret_analysis_header set datetime_finish = %s where job_id = %s"

# Executing query
cursor.execute(sql, (datetime.now(),i_process_id) )

print(i_process_id)

connection.commit()
connection.close()

print('job finished')


# ### 4. Marking as finished Job

# In[16]:


# Connect to the database
connection = pymysql.connect(host='202.157.176.225',
                             user='cekmedsos_db',
                             password='kuku838485*#',
                             database='cekmedsos_database',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

cursor = connection.cursor()

# inserting jobs into table mv result
# sql = "call spInsertResultToMV(%s);" 

# Executing query
# cursor.execute(sql,i_process_id)

sql = "update screen_analisis_ai set processby_id = 1, status = 3, end_process = now(), duration = TIMESTAMPDIFF(second,start_process, end_process) where id = %s"
# Executing query
cursor.execute(sql,i_process_id)

# commit changes
connection.commit()
connection.close()

print('inserting result finished')


# ### Wait 10 sec before querying next jobs

# In[ ]:


# Wait 10 sec before release
time.sleep(10)

