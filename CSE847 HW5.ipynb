from sklearn.decomposition import PCA
from scipy.io import loadmat

USPS = loadmat('USPS.mat')
A = USPS['A']
A.shape

p = [10, 50, 100, 200]

Error = []
for p_num in p:
    pca = PCA(n_components = p_num)
    p_comp = pca.fit_transform(A) #Fit model and apply dimensionality reduction
    
    img_recover = pca.inverse_transform(p_comp) #Transform back to the original space
    
    error = np.mean((A - img_recover)**2)
    Error.append(error)
    
    plt.figure(figsize=(10,4))
    for i in range(2):
        plt.subplot(1,2,i+1)
        plt.imshow(img_recover[i].reshape(16,16).T, cmap="gray")
        plt.title(f'Reconstructed Image {i+1} when p = {p_num}')
        plt.axis('off')
    plt.show()

for i, p_num in enumerate(p):
    print(f"When p={p_num}, the total reconstruction error is {Error[i]}")
