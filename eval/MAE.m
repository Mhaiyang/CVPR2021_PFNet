function mae = MAE(smap, gtImg)
if size(smap, 1) ~= size(gtImg, 1) || size(smap, 2) ~= size(gtImg, 2)
    error('Predicted map and gt Image have different sizes!\n');
end

if ~islogical(gtImg)
    gtImg = gtImg(:,:,1) > 128;
end

fgPixels = smap(gtImg);
fgErrSum = length(fgPixels) - sum(fgPixels);
bgErrSum = sum(smap(~gtImg));
mae      = (fgErrSum + bgErrSum) / numel(gtImg);