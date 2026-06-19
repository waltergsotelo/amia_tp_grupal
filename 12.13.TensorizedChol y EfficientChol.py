class TensorizedChol(QDA_Chol3):
    def _fit_params(self, X, y):
        # Invocamos el método de ajuste de la clase base QDA_Chol3 para obtener L_inv y las medias de cada clase
        super()._fit_params(X, y)
        # Apilamos las inversas de los factores de Cholesky (L_invs) en un tensor único de dimensiones (k, p, p)
        # donde k es la cantidad de clases y p es la cantidad de features/predictores
        self.tensor_L_invs = np.stack(self.L_invs)
        # Apilamos los vectores de medias (means) en un tensor de dimensiones (k, p, 1)
        self.tensor_means = np.stack(self.means)

    def predict(self, X):
        # Calculamos X centrado (sin la media) restando las medias de cada clase con broadcasting
        # Introducimos una dimensión extra al inicio de X para poder operar con las medias de todas las clases
        unbiased_X = X[np.newaxis, :, :] - self.tensor_means
        
        # Calculamos el término transformado Y aplicando la inversa del factor triangular
        # Y tiene forma (k, p, n) y representa la proyección de las observaciones centradas
        Y = self.tensor_L_invs @ unbiased_X
        
        # Calculamos la matriz de productos internos
        # Esto paraleliza el cálculo sobre todas las observaciones, pero genera una matriz intermedia grande de k x n x n
        # (esto es ineficiente en memoria y tiempo para n grande)
        M = Y.transpose(0, 2, 1) @ Y
        
        # Extraemos la diagonal de la matriz de productos internos para cada clase, obteniendo el término cuadrático
        # Corresponde a la norma L2 al cuadrado de cada columna
        quad_term = np.diagonal(M, axis1=1, axis2=2)
        
        # Calculamos el logaritmo del determinante del factor de Cholesky
        # Dado que L_inv es una matriz triangular, el log-determinante es la suma de los logaritmos de los elementos de su diagonal
        diags = self.tensor_L_invs.diagonal(axis1=1, axis2=2)
        log_det = np.sum(np.log(diags), axis=1)
        
        # Calculamos la probabilidad log a posteriori (más una constante) para cada combinación clase-observación
        # Sumamos la probabilidad log a priori, el log-determinante y restamos la mitad del término cuadrático
        log_posteriori = self.log_a_priori[:, np.newaxis] + log_det[:, np.newaxis] - 0.5 * quad_term
        
        # Seleccionamos la clase con la máxima probabilidad log a posteriori para cada observación
        y_hat = np.argmax(log_posteriori, axis=0)
        
        # Retornamos las predicciones adaptadas como un vector fila (1, n)
        return y_hat.reshape(1, -1)



class EfficientChol(QDA_Chol3):
    def _fit_params(self, X, y):
        # Invocamos el método de ajuste de la clase base QDA_Chol3 para obtener L_inv y las medias de cada clase
        super()._fit_params(X, y)
        # Apilamos las inversas de los factores de Cholesky (L_invs) en un tensor de forma (k, p, p)
        self.tensor_L_invs = np.stack(self.L_invs)
        # Apilamos las medias de cada clase en un tensor de forma (k, p, 1)
        self.tensor_means = np.stack(self.means)

    def predict(self, X):
        # Calculamos X centrado sin las medias de cada clase usando broadcasting
        unbiased_X = X[np.newaxis, :, :] - self.tensor_means
        
        # Proyectamos las observaciones centradas en el espacio transformado
        Y = self.tensor_L_invs @ unbiased_X
        
        # Calculamos de manera eficiente el término cuadrático de interés sin construir la matriz gigante k x n x n
        # Elevamos al cuadrado cada componente de Y y sumamos a lo largo del eje de las features.
        # Esto calcula directamente la norma L2 al cuadrado de cada columna, reduciendo la complejidad espacial
        quad_term = np.sum(Y ** 2, axis=1)
        
        # Calculamos el log-determinante sumando el logaritmo de los elementos de la diagonal
        diags = self.tensor_L_invs.diagonal(axis1=1, axis2=2)
        log_det = np.sum(np.log(diags), axis=1)
        
        # Calculamos la probabilidad log a posteriori para cada clase y observación
        log_posteriori = self.log_a_priori[:, np.newaxis] + log_det[:, np.newaxis] - 0.5 * quad_term
        
        # Elegimos la clase que maximiza la probabilidad log a posteriori para cada observación
        y_hat = np.argmax(log_posteriori, axis=0)
        
        # Retornamos el vector de predicciones como vector fila (1, n)
        return y_hat.reshape(1, -1)

