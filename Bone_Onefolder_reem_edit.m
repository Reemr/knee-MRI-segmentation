%calculate cartilage sampling
fclose all;
close all;
clear all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%update infomation
version = '1.0';
Dir_seg = 'C:\Users\Reem\Documents\51-60\9633944\v00\'; %%%%% change this to your local directory
file_name = '9633944_v00_TibiaBone'; %%%%change this to the txt file generated by the software for manual labeling result
ScanDirection = 1; %LEFT kneeMedial-->Lateral:1, Lateral-->Medial: 2 %OAI 1;IACS 2; David 1

%%%%% update voxel size here
%Voxel = 0.28*0.37*2.5;%BU/TMC
%OAI FS
% Voxel_x = 0.357;
% Voxel_y = 0.511;
% Voxel_z = 3;
 
%Voxel = 0.27; %VitD

%Voxel = 0.365*0.456*0.7 %OAI DESS
Voxel_x = 0.365;
Voxel_y = 0.456;
Voxel_z = 0.7;
%Voxel = 0.28*0.28*0.30 %David

Voxel = Voxel_x*Voxel_y*Voxel_z;

%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%IACS
% BVF (3D FFE) = 0.23 * 0.23 * 1 mm (this is the acquisition voxel size)
% BML/Effusion (2D PDW HR SPAIR) = 0.28 * 0.37 * 2.5 mm 
% CDI (3D WATSc SENSE) = 0.31 * 0.31 * 1 mm

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%create excel file
% if(exist('BML.xls','file'))
%     delete(strcat(Dir_seg,'BML.xls'));
% end

%}

 txt_filename = dir(Dir_seg);
 
%{ 
% 
% exl_title={'id','version','voxelx','voxely','slicethick','time','knee','start_slice', 'end_slice','bone','Medial_BML','Lateral_BML'};
% if(~xlswrite(strcat(Dir_seg,'BML.xls'),exl_title,'main','A1'))
%     disp('Cannot create excel file');
% end

%}


num = 0;
for i=3:length(txt_filename)
    %disp(patient_names(i).name);
    if(isempty(strfind(txt_filename(i).name,file_name)))
        continue;
    end
    if(isempty(strfind(txt_filename(i).name,'.txt')))
        continue;
    end
    %initialize BVF
    femur_part1 = 0;
    femur_part2 = 0;
    tibia_part1 = 0;
    tibia_part2 = 0;
    
    num = num+1;
    %read in effusion result file
    fid = fopen(strcat(Dir_seg,txt_filename(i).name),'r');
    id = fgetl(fid);
    disp(id);
    stage = fgetl(fid);
    
 %{   
%     if(~isempty(strfind(stage,'vET'))||~isempty(strfind(stage,'VET')))%<<---update here
%         stage_id = -100;
%     else
%         stage_id = str2double(stage(2:3));
%     end
    
%     if(~isempty(strfind(stage,'v00'))||~isempty(strfind(stage,'V00')))%<<---update here
%         stage_id = 1;
%     elseif(~isempty(strfind(stage,'v03'))||~isempty(strfind(stage,'V03')))%<<---update here
%         stage_id = 2;
%     elseif(~isempty(strfind(stage,'v24'))||~isempty(strfind(stage,'V24')))%<<---update here
%         stage_id = 3;
%     end

 %}

    kneeDirection = fgetl(fid);%LEFT or RIGHT knee
    fgetl(fid);
    sizeX = str2double(fgetl(fid)); % image size
    sizeY = str2double(fgetl(fid));
    start_slice = str2double(fgetl(fid));
    end_slice = str2double(fgetl(fid));
    result_femur = zeros(1,168);
    result_tibia = zeros(1,168);
    middle_slice = uint8((start_slice+end_slice)/2);
   
    %Femur
    if(~strcmp(fgetl(fid),'Femur'))
        disp('Missing Femur mark');
    end
    
    line = fgetl(fid);
    

    while(~strcmp(line,'Tibia'))
        slice_num = str2double(line);
        img = zeros(sizeX,sizeY);
        fgetl(fid);%threshold
        if(~strcmp(fgetl(fid),'{'))
            disp('Missing {');
        end
        line = fgetl(fid);
       
        BML=0;
        while(~strcmp(line,'}'))
            if(strfind(line,'.')) %boundary line
                space = strfind(line,' ');
                x=str2double(line(1:space(1)));
                dot = strfind(line,'.');
                for k=1:length(dot)%get x axles
                    y1 = str2double(line(space(k)+1:dot(k)-1));
                    y2 = str2double(line(dot(k)+1:space(k+1)-1));
                    img(sizeX-x,y1:y2)=255;
                    if(slice_num<=middle_slice)
                        femur_part1 = femur_part1+(y2-y1+1)*Voxel;
                    else
                        femur_part2 = femur_part2+(y2-y1+1)*Voxel;
                    end
                    BML=BML+(y2-y1+1)*Voxel;
                end   
            end
            line = fgetl(fid);
        end
        if(~exist(strcat(Dir_seg,id,'\',stage,'\'),'dir'))
            mkdir(strcat(Dir_seg,id,'\',stage,'\'));
        end
        %changed it to mask names 
        imwrite(img,strcat(Dir_seg,id,'\',stage,'\',id,'_',int2str(slice_num),'_mask.tiff'),'tiff');
        line = fgetl(fid); 
        result_femur(1,8+slice_num)=BML;
    end
    
    %Tibia
    if(~strcmp(line,'Tibia'))
        disp('Missing Tibia mark');
    end
    
    line = fgetl(fid);
    

    while(~strcmp(line,'Boneline'))
        slice_num = str2double(line);
        fgetl(fid);%threshold
        if(~strcmp(fgetl(fid),'{'))
            disp('Missing {');
        end
        line = fgetl(fid);
        BML=0;
        while(~strcmp(line,'}'))
            if(strfind(line,'.')) %boundary line
                space = strfind(line,' ');
                dot = strfind(line,'.');
                for k=1:length(dot)%get x axles
                    y1 = str2double(line(space(k)+1:dot(k)-1));
                    y2 = str2double(line(dot(k)+1:space(k+1)-1));
                    if(slice_num<=middle_slice)
                        tibia_part1 = tibia_part1+(y2-y1+1)*Voxel;
                    else
                        tibia_part2 = tibia_part2+(y2-y1+1)*Voxel;
                    end
                    BML=BML+(y2-y1+1)*Voxel;
                    
                end   
            end
            line = fgetl(fid);
        end
        line = fgetl(fid);   
        result_tibia(1,8+slice_num)=BML;
    end
    
    
    
    %write to excel file
    if (ScanDirection ==1) %Medial-->Lateral
        if(strcmp(kneeDirection,'LEFT'))
            femur_medial = femur_part1;
            femur_lateral = femur_part2;
            tibia_medial = tibia_part1;
            tibia_lateral = tibia_part2;
        elseif(strcmp(kneeDirection,'RIGHT'))
            femur_medial = femur_part2;
            femur_lateral = femur_part1;
            tibia_medial = tibia_part2;
            tibia_lateral = tibia_part1;
            
        else
            disp('Missing left or right knee information');    
        end
    elseif(ScanDirection ==2) %Lateral-->Medial
        if(strcmp(kneeDirection,'RIGHT'))
            femur_medial = femur_part1;
            femur_lateral = femur_part2;
            tibia_medial = tibia_part1;
            tibia_lateral = tibia_part2;
        elseif(strcmp(kneeDirection,'LEFT'))
            femur_medial = femur_part2;
            femur_lateral = femur_part1;
            tibia_medial = tibia_part2;
            tibia_lateral = tibia_part1;
            
        else
            disp('Missing left or right knee information');    
        end
    end
    if(strcmp(kneeDirection,'LEFT'))
        knee=2;
    else
        knee=1;
    end
    
    %write femur

%{    
%    % result_femur(1,1:8)=[str2double(id),stage_id,knee,start_slice,end_slice,1,femur_medial,femur_lateral];
%     exl_row = {id,version,Voxel_x,Voxel_y,Voxel_z,stage,knee,start_slice,end_slice,1,femur_medial,femur_lateral};
%    % A=[str2double(id),stage_id,femur_medial,femur_lateral,tibia_medial,tibia_lateral];
%     xlswrite(strcat(Dir_seg,'BML.xls'),exl_row,'main',strcat('A',int2str(num+1)));
%     num=num+1;
%     %write tibia
%     %result_tibia(1,1:8)=[str2double(id),stage_id,knee,start_slice,end_slice,2,tibia_medial,tibia_lateral];
%     exl_row = {id,version,Voxel_x,Voxel_y,Voxel_z,stage,knee,start_slice,end_slice,2,tibia_medial,tibia_lateral};
%     xlswrite(strcat(Dir_seg,'BML.xls'),exl_row,'main',strcat('A',int2str(num+1)));

%}
end



fclose all;