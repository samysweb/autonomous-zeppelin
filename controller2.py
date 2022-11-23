def control(x1, x2, o1, o2, c, v1, v2, p, r, T):
	vnorm = (v1**2 + v2**2)**0.5
	a1 = (x1 - o1)
	a2 = (x2 - o2)
	if a1*(v1/vnorm) + a2*(v2/vnorm) - c > 0:
		# Zeppelin is in slipstream: just follow the wind...
		y1 = v1/vnorm
		y2 = v2/vnorm
	else:
		q1 = (-c/(p - r)*v1)
		q2 = (-c/(p - r)*v2)
		qSquared = (q1**2 + q2**2)
		factor = (q1**2 + q2**2 - c**2)**0.5
		if a1*((-v1)/vnorm) + a2*((-v2)/vnorm) -(c/(p - r)*vnorm + T*r) > 0:
			# Zeppelin is flying towards Bermuda triangle: Move to respective side of triangle
			if -v1*a2 + v2*a1 > 0:
				y1 = (c*q1 + factor*(-q2))/qSquared
				y2 = (c*q2 + factor*q1)/qSquared
			else:
				y1 = (c*q1 - factor*(-q2))/qSquared
				y2 = (c*q2 - factor*q1)/qSquared
		else:
			if c*(q1*(a1 - q1) + q2*(a2 - q2)) + factor*(q1*a2 - q2*a1) > 0.0:
				# Zeppelin is on side 1 of Bermuda triangle: Move away
				y1 = (c*q1 + factor*(-q2))/qSquared
				y2 = (c*q2 + factor*q1)/qSquared
			else:
				if c*(q1*(a1 - q1) + q2*(a2 - q2)) - factor*(q1*a2 - q2*a1) > 0.0:
					# Zeppelin is on side 2 of Bermuda triangle: Move away
					y1 = (c*q1 - factor*(-q2))/qSquared
					y2 = (c*q2 - factor*q1)/qSquared
				else:
					# Zeppelin is inside Bermuda triangle:
					# Let's work against the wind and hope for the best...
					y1 = -v1/vnorm
					y2 = -v2/vnorm
	return (y1, y2)