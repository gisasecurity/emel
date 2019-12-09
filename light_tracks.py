import pandas as pd
# Panda untuk read write spreadsheet excel 
import numpy as np
# Numpy kalkulasi nilai matematik
import matplotlib.pyplot as pltfrom sklearn.cluster 
# Matplotlib untuk visualisasi data, chart
import KMeans from sklearn.preprocessing 
# Kmeans untuk kalkulasi dan pemikir , pemilah 
import MinMaxScaler
# MinMaxScaler untuk perhitungan skala.


# read data dari csv
light = pd.read_csv("light_tracks.csv")
light.head()

# filter kolom yang di butuhkan saja, drop artinya di hapus dari memory dan view
light = light.drop(["car_or_bus","time"], axis = 1)
light.head()

# Menentukan variabel yang akan di klusterkan ---
light_x = light.iloc[:, 1:3]
light_x.head()

# Menampilkan spreading atau persebaran data secara chart point
plt.scatter(light.distancetoground, light.speed, s =10, c = "c", marker = "o", alpha = 1)
plt.show()

# Mengubah Variabel Data Frame Menjadi Array 
x_array =  np.array(light_x)
print(x_array)
# Menstandarkan Variabel size sesuai match standar matematika common 
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_array)
x_scaled

# Menentukan dan mengeset konfigurasi fungsi daripada kmeans
kmeans = KMeans(n_clusters = 3, random_state=123)
# Menentukan kluster dari data
kmeans.fit(x_scaled)
# Menampilkan cluster center atau penyempitan prediksi  ---
print(kmeans.cluster_centers_)
# Menampilkan Hasil Kluster ---
print(kmeans.labels_)
# Menambahkan Kolom "kluster" Dalam Data Frame light ---
light["kluster"] = kmeans.labels_

# Memvisualkan hasil kluster ---
output = plt.scatter(x_scaled[:,0], x_scaled[:,1], s = 100, c = light.kluster, marker = "o", alpha = 1, )centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='red', s=200, alpha=1 , marker="s");
plt.title("Hasil Clustering berdasarkan analisis K-Means")
plt.colorbar (output)plt.show()
