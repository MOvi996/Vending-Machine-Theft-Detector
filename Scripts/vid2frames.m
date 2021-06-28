viddir=dir('D:\CV Project Data\*.mp4');
videos={viddir.name};

 
for i=1:numel(videos)
    aa=VideoReader(videos{i});

    for img = 1:5:aa.NumberOfFrames
        filename=strcat('what',num2str(i),num2str(img),'.jpg');
        folder='D:\CV Projects\'
        b = read(aa, img);
%     imshow(b);
        imwrite(b,strcat(folder,filename));
    end
end

