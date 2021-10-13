function angle = angle3D_deg( vec1, vec2 )
% ANGLE3D_DEG Returns the smallest angle between the
% three dimensional vectors 'vec1' and 'vec2'
% 
% NOTE: angle is returned in degrees

angle	=	180/pi * atan2(	norm( cross( vec1(:), vec2(:) ) ),  ...
                            dot( vec1(:), vec2(:) )	);

end

