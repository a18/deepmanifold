function deepart_pca(X,k)

data3=h5read('data_conv3_1.h5','/DS');
data4=h5read('data_conv4_1.h5','/DS');
data5=h5read('data_conv5_1.h5','/DS');

data3=reshape(data3,[size(data3,1)*size(data3,2)*size(data3,3) size(data3,4)]);
data4=reshape(data4,[size(data4,1)*size(data4,2)*size(data4,3) size(data4,4)]);
data5=reshape(data5,[size(data5,1)*size(data5,2)*size(data5,3) size(data5,4)]);

[U,T,mu]=pcaecon([data3; data4; data5],size(data5,2));
%[U,T,mu]=pcaecon([data4; data5],size(data5,2));

h5create('pca_U.h5','/DS',size(U),'Datatype','single','ChunkSize',[size(data5,1), 1]);
h5write('pca_U.h5','/DS',U);
h5create('pca_T.h5','/DS',size(T),'Datatype','single');
h5write('pca_T.h5','/DS',T);
h5create('pca_mu.h5','/DS',size(mu),'Datatype','single');
h5write('pca_mu.h5','/DS',mu);

