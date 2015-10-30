test_modulus = 10;

load ../pnas_split.mat;
path('../',path);

T = length(pnas_split);
ALL_test = [];
ALL_train = [];
for t = 1:T
  test_filename = ['T' num2str(t) '_test.dat'];
  train_filename = ['T' num2str(t) '_train.dat'];
  
  N_epoch = size(pnas_split{t}.data,1);
  test_idx = 1:test_modulus:N_epoch;
  test_data = pnas_split{t}.data(test_idx,:);
  train_data = pnas_split{t}.data;
  train_data(test_idx,:) = [];
  ALL_test = [ALL_test ; test_data];
  ALL_train = [ALL_train ; train_data];
  export_sparse_matrix(test_data,test_filename);
  export_sparse_matrix(train_data,train_filename);
end

ALL_test_filename = 'ALL_test.dat';
ALL_train_filename = 'ALL_train.dat';
export_sparse_matrix(ALL_test,ALL_test_filename);
export_sparse_matrix(ALL_train,ALL_train_filename);
