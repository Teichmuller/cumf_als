import math
# configurations
type_name = 'dtype'

reg_prefix = 'temp'
smem_name = 'thetaTemp'
rank_name = 'f'
iter_var_name = 'k'
array_name = 'array'
tile_x_name = 'tile_x'
tile_y_name = 'tile_y'
array_offset_name = 'tile_offset'

tile_dim = 10
file_name = 'reg_ops.h'

# initialize
tile_size = tile_dim * tile_dim
file_header = \
"""/*
 * reg_ops.h
 *
 * This is file is automatically generated. Please do not modify unless you know what you are doing.
 *
 *  Created on: Jul 19, 2017
 *      Author: Xuan
 */

#ifndef REG_OPS_H_
#define REG_OPS_H_

"""
file_end = \
"""
#endif /* REG_OPS_H_ */
"""
def get_reg_name(n):
	return reg_prefix + str(n)

fout = open(file_name, 'w')
fout.write(file_header)

# generate directives
text = ''
text += '#define RegisterTileSize ' + str(tile_dim) + '\n'
fout.write(text)

# generate declare_registers
text = ''
text += """
#define declare_registers()"""
for iter in range(tile_size):
	if iter % tile_dim == 0:
		if iter != 0:
			text += ';'
		text += '\\\n    ' + type_name + ' '
	text += get_reg_name(iter) + ' = 0'
	if iter % tile_dim != tile_dim - 1:
		text += ', '
text += '\n'
fout.write(text)

# generate accumulate_in_registers
text = ''
text += """
#define accumulate_in_registers()\\
    do\\
    {\\
"""
for x in range(tile_dim):
	for y in range(tile_dim):
		text += '        ' + get_reg_name(x * tile_dim + y) + ' '
		text += '+= ' + smem_name + '[' + tile_x_name + ' / 2 + '
		text += str(math.floor(x / 2))
		text += ' + ' + iter_var_name + ' * ' + rank_name + ' / 2].'
		if x % 2 == 0:
			text += 'x'
		else:
			text += 'y'
		text += ' * ' + smem_name + '[' + tile_y_name + ' / 2 + '
		text += str(math.floor(y / 2))
		text += ' + ' + iter_var_name + ' * ' + rank_name + ' / 2].'
		if y % 2 == 0:
			text += 'x'
		else:
			text += 'y'
		text += ';\\\n'
text += '    }\\\n    while (0)\n'
fout.write(text)

# generate fill_upper_half_from_registers
text = ''
text += """
#define fill_upper_half_from_registers()\\
    do\\
    {\\
"""
for x in range(tile_dim):
	for y in range(tile_dim):
		text += '        '
		text += array_name + '[' + array_offset_name + ' + ' + tile_x_name + ' + ' + str(x)
		text += ' + (' + tile_y_name + ' + ' + str(y) + ') * ' + rank_name + '] = '
		text += get_reg_name(x * tile_dim + y) + ';\\\n'
text += '    }\\\n    while (0)\n'
fout.write(text)

# generate fill_lower_half_from_registers
text = ''
text += """
#define fill_lower_half_from_registers()\\
    do\\
    {\\
"""
for x in range(tile_dim):
	for y in range(tile_dim):
		text += '        '
		text += array_name + '[' + array_offset_name + ' + ' + tile_y_name + ' + ' + str(y)
		text += ' + (' + tile_x_name + ' + ' + str(x) + ') * ' + rank_name + '] = '
		text += get_reg_name(x * tile_dim + y) + ';\\\n'
text += '    }\\\n    while (0)\n'
fout.write(text)

# close file
fout.write(file_end)
fout.close()
