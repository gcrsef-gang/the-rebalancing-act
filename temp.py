import geopandas as gpd

x = gpd.read_file("good_target_geometry.json")
post = gpd.read_file("post_target_geometry.json")
x = x.set_index("geoid")
post = post.set_index("geoid")
x_set = set(x.index)
post_set = set(post.index)
diff = post_set.difference(x_set)
geoid = post.loc[diff]
print(geoid)