function output_data = mtimes(self, input_data)
  % The language here assumes you have a higher than 3D trajectory
  % and a higher than 4D input_data. Of course this still works with just
  % a normal 2D input and 2D trajectory, but it takes a little thinking to see
  % how that works. The key is to think about how the 3+ dimensions of
  % trajectory are supposed to contain all *unique* slices
  % but the 4+ dimensions of the input data are supposed to contain all the
  % redundant dimensions to copy the transform onto.

  % Figure out needed 4D shapes
  size_arr = size(input_data);
  if length(size_arr) == length(self.kSize)
    input_data_z_size = 1;
  elseif length(size_arr) < 4
    input_data_z_size = size_arr(3);
  else
    input_data_z_size = prod(size_arr(4:end));
  end
  k_shape = [self.kSize_x, self.kSize_y, self.kSize_z, input_data_z_size];
  im_shape = [self.imSize_x, self.imSize_y, self.kSize_z, input_data_z_size];

  % squish input_data down to 4D
  if self.adjoint
    input_data = reshape(input_data, k_shape);
    output_data = zeros(im_shape);
  else
    input_data = reshape(input_data, im_shape);
    output_data = zeros(k_shape);
  end

  % Since z contains unique trajectories iterate through it to get 3D
  for i = 1:self.kSize_z
    % squish to 3D
    data_slice = squeeze(input_data(:,:,i,:));
    nufft_struct = self.nufft_structs(i);
    
    % Use 2D slices of the 3D input data
    for j = 1:input_data_z_size
      slice_2D = data_slice(:,:,j);
      if self.adjoint
        slice_2D = slice_2D(:).*self.w(:);
        res = nufft_adj(slice_2D, nufft_struct)/sqrt(prod(self.imSize));
      else
        res = nufft(slice_2D, nufft_struct)/sqrt(prod(self.imSize));
        res = reshape(res, self.kSize_x, self.kSize_y);
      end
      output_data(:,:,i,j) = res;
    end
  end
  % squeeze in case of singleton dimensions
  output_data = squeeze(output_data);
  % reset adjoint
  self.adjoint = 0;
end
