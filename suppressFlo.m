function [R,L] = suppressFlo(R,L,thresh,new_vals)
	% R=rand(240,320);
	% L=rand(240,320);
	% thresh=0.8;
	mag=((R.^2)+(L.^2)).^0.5;
	% size(mag)
	bin=mag<thresh;
	% size(bin)
	% sum(sum(bin))
	
	R(bin)=new_vals(1);
	L(bin)=new_vals(2);

	% mag=((R.^2)+(L.^2)).^0.5;
	% size(mag)
	% bin=mag<thresh;
	% size(bin)
	% sum(sum(bin))
	

	% 240*320
	% max(mag)
end