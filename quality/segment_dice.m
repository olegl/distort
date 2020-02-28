% the darker parts are of interest!
%ground = double(imread('crop_eval_ground_0018.png'));
%reg = double(imread('crop_eval_reg-rigid_0018.tiff'));
ground = double(crop500_out1000);
reg = double(crop500_out1001);

threshold = 100;

mask_g = ground.*(ground<threshold);
mask_r = reg.*(reg<threshold);

bw_g = activecontour(ground,mask_g,300);
bw_r = activecontour(reg,mask_r,300);

similarity = dice(bw_g, bw_r);


%imshow(bw_g)
%imshow(bw_r)

imwrite(bw_g, '/raid5/sda1/AGCM/CT_ground/proc/CT/Funatomi_20200208/eval_1k/crop_T100_eval_Funatomi_matlab_mask_1000.png')
imwrite(bw_r, '/raid5/sda1/AGCM/CT_ground/proc/CT/Funatomi_20200208/eval_1k/crop_T100_eval_Funatomi_matlab_mask_1001.png')

figure
imshowpair(bw_g, bw_r);
title(['Dice Index = ' num2str(similarity)])
saveas(gcf, '/raid5/sda1/AGCM/CT_ground/proc/CT/Funatomi_20200208/eval_1k/crop_T100_eval_Funatomi_matlab_DICE_1000-1001.pdf')
