import numpy as np


class KMeansClustering:
    def __init__(self, n_clusters = 3):
        self.k = n_clusters
        self.centroids = None
    
    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))

    def fit(self, data, max_iterations = 200):
        # gerando os centróides dentro da imagem de forma aleatória
        self.centroids = np.random.uniform(np.amin(data, axis=0),
                                           np.amax(data, axis=0),
                                           size=(self.k, data.shape[1]))
        
        for _ in range(max_iterations):
            y = []

            # calculando a distancia entres os pontos e os centróides
            for data_point in data:
                # caculando a distancia euclidiana para cada centróide
                distances = self.euclidean_distance(data_point, self.centroids)
                # procurando a menor distância dentre os centróides e retornando o seu index
                cluster_num = np.argmin(distances)
                # guardando o indice da menor distância
                y.append(cluster_num)
            # transformando a array em uma array do numpy
            y = np.array(y)

            # preparando para a próxima avaliação
            cluster_indices = []

            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))
            
            # recentralizando os clusters
            cluster_centers = []
            for i, indices in enumerate(cluster_indices):
                # caso o número de clusters e k sejam diferentes podem existir clusters sem membros atribuídos
                if len(indices) == 0:
                    # neste caso eles não serão recentralizados
                    cluster_centers.append(self.centroids[i])
                else:
                    # caso o cluster tenha membros atribuidos a média das posições será o novo cluster
                    cluster_centers.append(np.mean(data[indices], axis=0)[0])
            # se a distância máxima entre os centróides novos e antigos não for significativa não é necessário fazer mais iterações
            if np.max(self.centroids - np.array(cluster_centers)) < 0.001:
                break
            else:
                # caso contrario os clusters são reposicionados
                self.centroids = np.array(cluster_centers)

        # retorna os labels de cada cluster
        return y
