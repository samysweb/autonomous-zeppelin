def control(x1, x2, o1, o2, c, v1, v2, p, r, T):
	a1 = x1 - o1
	a2 = x2 - o2
	vnorm = (v1**2 + v2**2)**0.5
	if a1*(v1/vnorm) + a2*(v2/vnorm) - c > 0:
		# Zeppelin is in slipstream of obstacle
		y1 = v1/vnorm
		y2 = v2/vnorm
	else:
		mu = (-c/(p - r))
		muv = ((mu*v1)**2 + (mu*v2)**2)
		muvnorm = ((mu*v1)**2 + (mu*v2)**2 - c**2)**0.5
		if -v1*a2 + v2*a1 > 0:
			# Zeppelin is on side 1 of obstacle
			y1 = (c*(mu*v1) + muvnorm*(-c/(p - r)*(-v2)))/muv
			y2 = (c*(mu*v2) + muvnorm*(mu*v1))/muv
		else:
			# Zeppelin is on side 2 of obstacle
			y1 = (c*(mu*v1) - muvnorm*(-c/(p - r)*(-v2)))/muv
			y2 = (c*(mu*v2) - muvnorm*(mu*v1))/muv
	return (y1, y2)