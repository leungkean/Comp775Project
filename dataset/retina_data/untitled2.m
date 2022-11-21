myDir = '/Users/keanl/Desktop/Computer_Science/Comp_775/project/dataset/retina_data';
destDir = '/Users/keanl/Desktop/Computer_Science/Comp_775/project/dataset/retina_filter';
myFiles = dir(fullfile(myDir,'*.png'));
for k = 1:length(myFiles)
    I = imread(myFiles(k).name);
    I = imresize(I,[32, 32]);
%     I = rgb2gray(I);
%     I = imbinarize(I);
%     I = I(:,1:130);
%     I = imresize(I, 0.5);
    imwrite(I, fullfile(destDir, myFiles(k).name));
end
% I = imread('im0002.net.png');
% I = rgb2gray(I);
% I = imbinarize(I);
% I = I(:,10:140);
% I = imresize(I, 0.5);


