import file_handler
import k_means

path = 'Dataset1.csv'


if __name__ == '__main__':
    X, Y = file_handler.read_from_file(path)

    K = 4
    iterations = 30

    data = list(zip(X, Y))
    errors = []

    clusters, centers, data_from_cluster = \
                k_means.specifying_clusters(data, K, iterations)
    k_means.draw_data(data, centers, data_from_cluster, K, iterations)

    k_means.compute_error_for_some_k(data, K, iterations)
