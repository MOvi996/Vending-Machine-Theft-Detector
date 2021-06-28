clc;
close all;
cd D:\Ovais
loadnetfile2=load('D:\Ovais\Frames\weight 3 rand epoch 3');
net2= loadnetfile2.net;
% startingFolder = pwd;
%     defaultFileName = fullfile(startingFolder,...    
%     {'*.mp4;*.MP4;','Supported Files (*.mp4,*.MP4)'; ...
%     '*.mp4','Mp4 Files (*.mp4)';...
%     '*.MP4','MP4 Files (*.MP4)'});
    
%tstdir=dir('D:\Ovais\ovais videos\Testing videos\*.mp4');
%fNames={tstdir.name};    
foldert='D:\Ovais\ovais videos\Testing videos';
filetype='*.mp4';
f=fullfile(foldert,filetype);
dt=dir(f);
for j=1:numel(f)
    reader = VideoReader(fullfile(foldert,dt(j).name));

    writer = VideoWriter(strcat('videotst', num2str(j), 'weight 3.avi'));

    open(writer);
    position = [1 1]; 
    tic
    while hasFrame(reader)
        img = readFrame(reader);

        I=imresize(img,[224 224]);
        [label,score]=classify(net2,I);
  
        text_str = [char(label) ':' num2str(score())];
        img = insertText(img,position,text_str,'FontSize',20,'TextColor','red');
   
   
        writeVideo(writer,img);
    end
end
toc
close(writer);
