#to try locally call from cmd prompt: streamlit run streamlit_test1.py

import streamlit as st
import networkx as nx
from pyvis.network import Network
import json
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.special import comb
import requests
import base64
from collections import Counter

def url_to_base64(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        
        base64_data = base64.b64encode(response.content).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_data}"
    except Exception as e:
        print(f"Error: {e}")
        return None


def exploration_score(users_list, main_users, friends_list, remove_main = False):
    """Efficient user popularity / degree score as a dictionary."""
    
    # Count appearances of each user in friends_list
    counts = Counter()
    for a, b in friends_list:
        counts[a] += 1
        counts[b] += 1

    if remove_main:
        # Convert main_users to set for O(1) lookups
        main_set = set(main_users)
        # Build the score dictionary only for users not in main_users
        scores = {u: counts.get(u, 0) for u in users_list if u not in main_set}
    else:
        scores = dict(counts)

    return scores



def iterated_degree_voting_normalized(users_list, main_users, friends_list, iter=3, remove_main = False):

    ''' 
    Returns a score dictionary based on an iterated weigthed degree voting.
    At each iteration, the weight of the vote of a user is equal to his score.
    At the first iteration, the scores are 1 for all users
    At the second the score of a user becomes equal  to the degree of the user in the graph.
    From the third, the score is deterministic at each iteration.
    With a simple sum all the scores would grow exponentially over the iterations, 
    so a normalization by total is applied at each step.
    The final score can be converted to a theoretical degree 
    if multiplied by the number of nodes in the graph, since all the scores sum up to 1.

    users_list : list of strings (usernames)
    main_users : list of strings (usernames) 
    friends_list: list of tuples of two strings (usernames-nodes in the graph network)   

    '''

    scores = {u: 1 for u in set(users_list+main_users)}
    counts = Counter(scores)

    for i in range(iter):
        old_counts = counts.copy()
        total_old = old_counts.total()
        
        # Normalize old counts to prevent explosion
        normalized_old = {u: count/total_old for u, count in old_counts.items()}
        
        # Reset counts for new iteration
        counts = Counter()
        for a, b in friends_list:
            counts[a] += normalized_old.get(b, 0)
            counts[b] += normalized_old.get(a, 0)
        
        # Optional: Add small constant to avoid zero scores
        for u in set(users_list+main_users):
            counts[u] += 1e-8

    if remove_main:
        main_set = set(main_users)
        scores = {u: counts.get(u, 0) for u in users_list if u not in main_set}
    else:
        scores = dict(counts)
        
    return scores




# --- Load data ---
def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

#main_data = load_json('my_data.json')


# main_data = {
#     'friends_list':[('A', 'B'), ('B', 'C'), ('C', 'B'), ('A', 'F'), ('B', 'F'), ('C', 'E'), ('A', 'D')],
#     'users_info':{'A': {'username': 'andrew', 'biography' : 'hi',  'scraped_follower' : True},
#                   'B': {'username': 'bob', 'biography' : 'ho', 'scraped_follower' : False}
#                   },
#     'main_users': ['A', 'B'],
#     'users_ids' : ['A', 'B', 'C', 'D', 'E', 'F'],
#     'profilepic': {}
# }

if 'data' not in st.session_state:

    main_data = load_json('users_data.json')
    main_data['friends_list'] = load_json('edges_data.json')
    main_data['profilepic'] = load_json('images_data.json')

    st.session_state.data = main_data

main_data = st.session_state.data

if 'users_list' not in st.session_state:
    users_list = list(main_data['users_ids'].keys())
    st.session_state.users_list = users_list

users_list = st.session_state.users_list

if 'users_counts' not in st.session_state:
    important_users_counts = np.array([(main_data['users_info'][u]['scraped_followers'] or main_data['users_info'][u]['scraped_following']) for u in main_data['users_info']])
    main_users_count = important_users_counts.sum()
    interesting_users_count = important_users_counts.size - main_users_count

    st.session_state.users_counts = (main_users_count, interesting_users_count, main_users_count + interesting_users_count, len(main_data['users_ids']), len(main_data['friends_list']))

if 'users_groups' not in st.session_state:
    # unknown_u = []
    # inter_u = []
    # main_u = []

    # for u in users_list:
    #     if u in main_data['users_info']:
    #         if main_data['users_info'][u]['scraped_followers']:
    #             main_u.append(u)
    #         else:
    #             inter_u.append(u)
    #     else:
    #         unknown_u.append(u)
    users_set = set(users_list)
    known_users_set = set(main_data['users_info'].keys())

    # 1. Find all unknown users at once using set difference. This is the biggest speedup.
    unknown_u = list(users_set - known_users_set)

    # 2. Find users that need further checking using set intersection.
    # This loop will now be much shorter.
    users_to_check = users_set & known_users_set

    # 3. Initialize the final lists
    main_u = []
    inter_u = []

    # 4. Loop ONLY over the much smaller group of known users to categorize them.
    for u in users_to_check:
        # No need to check if u is in main_data['users_info'] again - we already know it is.
        if main_data['users_info'][u]['scraped_followers']:
            main_u.append(u)
        else:
            inter_u.append(u)

    st.session_state.users_groups = (main_u, inter_u, unknown_u)



main_users_count, interesting_users_count, important_users_count, tot_num_of_users, tot_num_of_friendships = st.session_state.users_counts


if 'global_degree' not in st.session_state:
    st.session_state.global_degree = exploration_score(users_list, main_data['main_users'], main_data['friends_list'])
    

# --- Streamlit UI ---
st.set_page_config(page_title="instaronno", layout="wide")
st.title("Instaronno")

st.markdown(
    f":red-badge[#main users : {main_users_count}]  :violet-badge[#interesting users : {interesting_users_count}] :orange-badge[#known users : {important_users_count}]  :blue-badge[#observed users : {tot_num_of_users}]    :orange-badge[#friendships (follower+following) : {tot_num_of_friendships}]"
)





########################################### Sidebar controls
st.sidebar.header("Graph Settings")


# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light☀︎'.lower()

# Theme selector - place this in your sidebar
with st.sidebar:
    theme = st.selectbox(
        "Select Theme",
        ["Light☀︎", "Dark⏾"],
        index=0 if st.session_state.theme == 'Light☀︎'.lower() else 1
    )




# Update theme
if theme.lower() != st.session_state.theme:
    st.session_state.theme = theme.lower()
    st.rerun()
# Comprehensive dark theme CSS

darktheme = (st.session_state.theme == 'Dark⏾'.lower())


if darktheme:
    dark_theme_css = """
    <style>
    /* Main app background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #0E1117;
    }
    
    /* All text elements */
    .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, div, span {
        color: #FAFAFA !important;
    }
    
    /* Selectbox main element */
    div[data-baseweb="select"] > div {
        background-color: #262730 !important;
        color: #FAFAFA !important;
        border-color: #515561 !important;
    }
    
    /* Selectbox dropdown popover */
    div[data-baseweb="popover"] {
        background-color: #262730 !important;
    }
    
    /* Selectbox dropdown menu */
    div[data-baseweb="menu"] {
        background-color: #262730 !important;
        color: #FAFAFA !important;
    }
    
    /* Selectbox dropdown list */
    ul[data-baseweb="menu"] {
        background-color: #262730 !important;
    }
    
    /* Selectbox dropdown items */
    li[role="option"] {
        background-color: #262730 !important;
        color: #FAFAFA !important;
    }
    
    /* Selectbox dropdown items - hover state */
    li[role="option"]:hover {
        background-color: #515561 !important;
        color: #FAFAFA !important;
    }
    
    /*Help content*/
    div[data-testid="stTooltipContent"] {
        background-color: #515561 !important;
        color: #FAFAFA !important;
    }

    /* Help icon*/
    svg[class="icon"]{
        background-color: #FAFAFA !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #262730 !important;
        color: #FAFAFA !important;
        border: 1px solid #515561 !important;
    }
    
    .stButton button:hover {
        background-color: #515561 !important;
        color: #FAFAFA !important;
        border-color: #71757c !important;
    }
    </style>
    """
    st.markdown(dark_theme_css, unsafe_allow_html=True)

with st.sidebar:
    show_help = st.checkbox('show help', False)
    show_graph = st.checkbox('show graph', True)
    show_graph_stats = st.checkbox('show graph stats', False)



st.sidebar.header('Nodes layout')

layout_dict = {
    "Fruchterman-Reingold" : nx.spring_layout,
    "Kamada-Kawai" : nx.kamada_kawai_layout,
    "circular" : nx.circular_layout,
    "shell" : nx.shell_layout,
    'arf': nx.arf_layout,
    'ForceAtlas2' : nx.forceatlas2_layout,
    'random': nx.random_layout,
    'spectral': nx.spectral_layout,
    'spiral': nx.spiral_layout
}

layout_choice = st.sidebar.selectbox(
    "Choose layout algorithm",
    list(layout_dict.keys()),
    help = """
    Select the layout algorithm to use 
    for the initial positioning of the nodes.
    If You activate the physics from the physics panel,
    that ovverrides this initial configuration.
    The options are described at [netwrox_draw](https://networkx.org/documentation/stable/reference/drawing.html#module-networkx.drawing.layout).

    Between the options:
    - random positions the nodes uniformly at random in the unit square
    - spiral builds a spiral
    - circular builds a circle
    - shell builds concentric circles
    - spectral uses the eigenvectors of the graph Laplacian
    - FR and KK are nice obscure algorithms
    - arf and ForceAtlas2 keep long distances between the nodes

    """
)


enable_physics = st.sidebar.checkbox('physics enabled', False)



########################################## Help paragraph


if show_help:
    st.markdown(
'''
## about
Welcome to Instaronno! 
The biggest database about the groups of Instagram users in the worderful Saronno city.
Here You can apply graph theory to some of the users whose data we have gathered
from the IG social. If You want to be scraped, please go to IG and follow the user lupinrosa.
This project is written in python, using the library instagrapi to gather the data, 
streamlit to build the web dashboard, networkx to make graph analysis and pyvis to have
the interactive graph dispalyer.
Both data and code are published on Github.

## help
In the sidebar, You have the main controls of the graph. They should be monkeyproof,
however some humans can be more stupid than the monkeys and don't succeed to understand it.
In this page, You have two main tools:
- the graph viewer, that lets observe directly the graph you have build
- the graph stats panel, that gives some statistics and plots that are commonly used to describe your graph.
If You want to use just one of them, is suggested to disable the other, since
both can require some time to load.
The sidebar contains the main controls for the graph viewer.

#### the sidebar
In the sidebar, under "Select central users", You can search and add to a list
the usernames of the users You absolutely want to have in your graph.
After each selection, to the graph are added other users, friends of your central users,
following one of the criteria You can choose from "nodes gathering criteria".
(other details on help button hover)
Pay attention to the option You select there.
If You have many central users added to Your list, maybe an economic gathering option is suggested,
such as 'only selected' or 'common friends of selected'.
The option 'all friends of selected' is suggested to be used only with one or two central users in your list,
when they have few friends in the data, otherwise the number of nodes and edges will
grow exponentially and your page will sure crash.
The option 'all data' is proposed as a trap, as educational purpose, so please don't select it except 
if You want to see how the page crash.

From the Styling section in the sidebar You can change many 
estetics parameters of the graph.
Under Node labels section You can select some users level informations 
that You can see when hovering the nodes, like their number of edges in the built graph (degree)
or the number of edges in the overall database (global degree, that should be less or equal to follower+following).
From the Advanced Settings section You can activate some 
built-in graph interaction tools that lets You play dynamically with the graph,
like the physics panel, the interaction panel, the filter and search menus.
Many of them will be inserted under the graph, inside the graph viewer.
On default, navigation buttons, keyboard controls and hovering reactions are enabled.
If physics is disabled, the static positions of the nodes are choose by the layout algorthm
under Graph settings.



#### users type and data collection algorithm
You can note that in the node styling is made a distinction between the colour
of the 'main users', the 'interesting users' and all the others.
The reason is the way in which the data are gathered from IG.
To gather the data we have used the instagrapi library for python,
that we have used to get two main types of informations for some users:
- the specific info of the user (such as full name, bio, number of follower,
has private, verified or business account etch)
- the list of the friends of the user

Unfortunately, while the first type of information is easy to get (around 4-5 seconds)
the second, so the list of the friends, requires quite a bit (not less than 3 minutes,
depending on how many followers and following the observed user has).
Furthermore, if a user has a private account, if is not friend of our bot scraper account,
we cannot see his friends list
So, to optimize the scraping procedure, we have assigned a local score to all the not scraped users,
based on the degree of them in the graph with all the data.
The not scraped user with the higher degree is selected as the next search,
and its specific data are gathered. If he has not a private account,
then we collect also his list of friends, and he becomes a main user.
If he has a private account, then we just tag him as an interesting user.
The web scraping procedure is executed locally, and the data You can see and analize in the app
are the more recent checkpoint of the process.

So we have made such distinction in our data:
- the main users are users for which we have both the users 
info and the list o his friends
- the interesting are users of which we have their users info 
but not their friends list, often because they have a private account
- the others are just usernames that we see in the friends list 
but we have not (already) tried to scrape, 
maybe beacause his global degree score is not enought comparing to others.

''')




# #### legal disclaimer
# All the data visible in this app are already published from IG.
# If You have find out to be inside it and You don't want to, write us and we will 
# immediately remove all your data from our db and prevent that they will be collected in future.
# If You think web scraping is illegal in Italy, please note
# that from 2021 the EU has allowed it for no profit and educational data mining activities.
# Is something has been violated, maybe it can be find in the terms of service of IG, the same
# that allow IG to sell your data to private companies.
# Lastly, please try always to find out what is morally right, particularly when the law is not clear or ambiguous.







############################################### select data

st.sidebar.header("Select central users")

filter_type = st.sidebar.selectbox(
    "nodes gathering criteria:",
    ["common friends of selected", "pairwise common friends",
     "all friends of selected","only selected", "all data (DANGER)"],
     help ="""
     Choose the way in which the nodes and the edges of the graph are selected from the data.
     This is based on some main nodes (users) selectable in the list below.
     THIS LIST IS ORDERED BY DECREASING COMPUTATIONAL COST
     
     - <only selected> gives only the nodes in list, 
     - <common friends of selected> gives them and each node that is friend (follower/following) of ALL of them,
     - <pairwise common friends> gives them and each node that is friend of AT LEAST TWO of them,
     - <all friends of selected> gives all friends of each node in list. It can be computationally expensive,
     especially when You select more than one user or a user You have selected has many friends.
     - <all data> is a stupid option, and You should be an idiot if You truly think your page won't crash when You select it.
     """
)




############################################# select users


with st.sidebar:
    # Simulate a large list
    #items = [f"Item {i}" for i in range(50000)]


    def update_items():
        st.session_state.updated = True

    def add_item(it):
        if it not in st.session_state.selected_items:
            st.session_state.selected_items.append(it) #members of the ui list
            st.session_state.bool_dict[it] = True      #members of the graph

    def add_only_to_graph(it):
        st.session_state.bool_dict[it] = True      #members of the graph


    if "myitems" not in st.session_state:
        st.session_state.myitems = sorted(users_list)   #list(main_data['users_ids'].keys())

    items = st.session_state.myitems

    # Session state
    if "selected_items" not in st.session_state:
        st.session_state.selected_items = []

    if 'bool_dict' not in st.session_state:
        st.session_state.bool_dict = {u : False for u in items}
        main_u, inter_u, unknown_u = st.session_state.users_groups
        firstusers = np.random.choice(main_u, 3).tolist() #+ np.random.choice(inter_u, 5).tolist() + np.random.choice(unknown_u, 10).tolist()
        for u in firstusers:
            add_item(u)
        #add_item('lorenzo_gallinaro')
        #add_item('gioele_carlini')
        #add_item('_martabanfi')

        #st.session_state.bool_dict[] = True
        #st.session_state.bool_dict[] = True

    # Search box + filtered results (limit for performance)
    #search = st.text_input("🔍 Search for IG username")

    # if search:
    #     results = [it for it in items if search.lower() in it.lower()][:30]
    # else:
    #     results = []

###################################### random collection

    def remove_all_random():
        G = st.session_state.G
        for u in G.nodes:
            if u not in st.session_state.selected_items:
                st.session_state.bool_dict[u] = False

    def remove_zero_degree():
        G = st.session_state.G
        degrees = dict(G.degree)
        for u in G.nodes:
            if degrees[u] == 0:
                if u in st.session_state.selected_items:
                    st.session_state.selected_items.remove(item)
                st.session_state.bool_dict[u] = False
            



    auto_collection = st.checkbox('USE NODES SAMPLER', False, 
help=f'''
Check to generate random graphs, randomly drawing nodes from the data.
''')

    if auto_collection:

        add_to_graph = st.checkbox('add directly to graph', True,
help = f'''
If checked the nodes are added directly to the graph. 
If unchecked they are added to the list of collected users.
Is suggested to uncheck this only when the 'nodes gathering criteria'
is set to <only selected>.
'''
)

        collection_method = st.selectbox('sampling method:',
        ['probability by degree', 'probability by global degree', 'probability by follower', 'probability by following','probability by follower+following', 'uniformly at random'],
help=f'''
When the button SAMPLE is pressed, the selected number of nodes/user
is sampled from the data.
The sampling options are mainly two:
- <uniformly at random> : draw nodes completely at random
- <probability by *> : draw nodes using a probability distribution given by the normalization of *.

Note that here degree is a popularity score given by the nodes ALREADY in the graph to all the other nodes,
while global degree is the degree of each node given by the main users in the database.
Follower, Following and their sum are other three values that we can convert to a normalized score
to obtain a probability distribution, however,
since their values are available only for the main and the interesting users,
the other users are excluded.
''')

        fixed_seed = st.checkbox('fix seed', 
help=f'''
A seed is a number that identifies 
the simulation result. 
You won't obtain new nodes from sampling
if You use the same seed more than once.
Check if You need to replicate exactly a selection of nodes
in a single sampling action.
''')
        if fixed_seed:
            sample_seed = st.slider('simulation seed', 1, 100, 42, 1)
            np.random.seed(sample_seed)

        include_main_u = st.checkbox('include main users', True)
        include_inter_u = st.checkbox('include interesting users', True)
        include_unknown_u = st.checkbox('include unknown users', True)

        num_to_collect = st.slider('num of users to sample:', 1, 100, 1, 1)

        add_at_random = st.button('SAMPLE')
        remove_random = st.button('remove all drawed nodes',
                                  help = 'remove sampled nodes')

        if add_at_random:

            main_u, inter_u, unknown_u = st.session_state.users_groups

            list_of_candidates = []

            if include_main_u:
                list_of_candidates += main_u
            if include_inter_u:
                list_of_candidates += inter_u
            if include_unknown_u and (collection_method not in ('probability by follower', 'probability by following','probability by follower+following')):
                list_of_candidates += unknown_u            
            if not list_of_candidates:
                list_of_candidates = users_list

            if collection_method == 'uniformly at random':
                sampled = np.random.choice(list_of_candidates, num_to_collect, replace=False)
            else:
                if collection_method == 'probability by global degree':
                    W = np.array([st.session_state.global_degree[u] for u in list_of_candidates])
                    sampled = np.random.choice(list_of_candidates, num_to_collect, replace=False, p=W/W.sum())
                else:
                    if collection_method == 'probability by follower':
                        W = np.array([main_data['users_info'][u]['followers_count'] for u in list_of_candidates])
                        sampled = np.random.choice(list_of_candidates, num_to_collect, replace=False, p=W/W.sum())
                    else:
                        if collection_method == 'probability by following':
                            W = np.array([main_data['users_info'][u]['following_count'] for u in list_of_candidates])
                            sampled = np.random.choice(list_of_candidates, num_to_collect, replace=False, p=W/W.sum())
                        else:
                            if collection_method == 'probability by follower+following':
                                W = np.array([main_data['users_info'][u]['followers_count']+main_data['users_info'][u]['following_count'] for u in list_of_candidates])
                                sampled = np.random.choice(list_of_candidates, num_to_collect, replace=False, p=W/W.sum())
                            else:
                                if collection_method == 'probability by degree':
                                    G = st.session_state.G
                                    scores = exploration_score(list_of_candidates, G.nodes, #list(G.nodes.keys()), 
                                                            main_data['friends_list'])
                                    W = np.array([scores[u] for u in list_of_candidates])
                                    sampled = np.random.choice(list_of_candidates, num_to_collect, replace=False, p=W/W.sum())
                                else:
                                    sampled = []

            if add_to_graph:
                for u in sampled:
                    st.session_state.bool_dict[u] = True
            else:
                for u in sampled:
                    add_item(u)


        if remove_random:
            remove_all_random()



##################################### search bar

    mysel = st.selectbox("🔍 Search for IG username", items, #results
                          on_change=update_items, placeholder='write something')
    st.button('add user', on_click=update_items)


############################ collected panel


    st.caption('🧺 Collected users ')


    if "updated" not in st.session_state:
        st.session_state.updated = False

    updated = st.session_state.updated
    if updated:
        st.session_state.updated = False
        add_item(mysel)


    # Create a copy of the list to avoid modifying while iterating
    items_to_remove = []

    with st.container(height = 100):
        for item in st.session_state.selected_items:
            col_btn, col_text = st.columns(2)
            with col_btn:
                # Use a unique key for each button
                if st.button("❌", key=f"remove_{item}"):#, help=f"Remove {item}"):
                    items_to_remove.append(item)
                    st.session_state.bool_dict[item] = False

            with col_text:
                st.write(item)

    # Remove items after the iteration is complete and trigger immediate rerun
    if items_to_remove:
        for item in items_to_remove:
            if item in st.session_state.selected_items:
                st.session_state.selected_items.remove(item)
        st.rerun()  # This ensures the UI updates immediately

    # Optional: Add a clear all button
    if st.session_state.selected_items:
        if st.button("Clear All"):
            remove_all_random()
            for u in st.session_state.selected_items:
                st.session_state.bool_dict[u] = False
            st.session_state.selected_items.clear()
            st.rerun()

    remove_lonely = st.button('remove lonely nodes',
                                help='removes nodes with degree = 0')

    if remove_lonely:
        remove_zero_degree()





# # 1. Search Bar
# search_query = st.sidebar.text_input("Search main users", placeholder="Type here to filter...",
# help="""
# Search and select one or more users from the data. They will be used by the users filter
# to apply the algorithm of node filtering and build the graph.
# """)

# # Your list of strings
# my_list = list(set(main_data['main_users']))  #users_list is too much

# if 'bool_dict' not in st.session_state:
#     st.session_state.bool_dict = {u : False for u in my_list}
#     st.session_state.bool_dict['lorenzo_gallinaro'] = True
#     st.session_state.bool_dict['gioele_carlini'] = True
    
# # 2. deselect all
# def uncheck_all():
#     st.session_state.bool_dict = {u : False for u in my_list}


# st.sidebar.button('uncheck all', on_click=uncheck_all)

# # Create a container with a fixed height
# with st.sidebar.container(height=150):

#     filtered_options = [
#     opt for opt in my_list if search_query.lower() in opt.lower()
#     ]

#     # Loop through the list to create a checkbox for each item
#     for item in filtered_options:
#         st.session_state.bool_dict[item] = st.checkbox(item, value=st.session_state.bool_dict[item])



##################################################### gather filtered data



def get_friends_in_club(club, friends_list):
    edges = [(a, b) for a, b in friends_list if (a in club) and (b in club)]
    return edges

def get_friends_of_club(club, friends_list):
    edges = [(a, b) for a, b in friends_list if (a in club) or (b in club)]
    commons = list(sum(edges, ()))

    list_of_friends = [(u,v) for (u,v) in friends_list if ((u in commons) and (v in commons))]

    return list_of_friends

def get_common_friends(user1, user2, friends_list):
    user1_friends = {b for a, b in friends_list if a == user1} | {a for a, b in friends_list if b == user1}
    user2_friends = {b for a, b in friends_list if a == user2} | {a for a, b in friends_list if b == user2}
    common = user1_friends.intersection(user2_friends)
    return list(common)

def pair_common_friends(club, friends_list):
    commons = []
    for a in club:
        for b in club:
            if a != b:
                commons.extend(get_common_friends(a, b, friends_list))

    list_of_friends = [(u,v) for (u,v) in friends_list if (((u in club) or (u in commons)) and ((v in club) or (v in commons)))]

    return list_of_friends

def get_club_unanimity(club, friends_list):
    user = club[0]
    commons = {b for a, b in friends_list if a == user} | {a for a, b in friends_list if b == user}
    for i in range(len(club)):
        user = club[i]
        user_friends = {b for a, b in friends_list if a == user} | {a for a, b in friends_list if b == user}
        commons = user_friends.intersection(commons)

    list_of_friends = [(u,v) for (u,v) in friends_list if (((u in club) or (u in commons)) and ((v in club) or (v in commons)))]

    return list_of_friends




bosses = [u for u in st.session_state.bool_dict.keys() if st.session_state.bool_dict[u]]
if bosses:
    nodes = bosses
    if filter_type == 'only selected':
        edges = get_friends_in_club(bosses, main_data["friends_list"])
    else:
        if filter_type == 'all friends of selected':
            edges = get_friends_of_club(bosses, main_data["friends_list"])
        else:
            if filter_type == 'pairwise common friends':
                edges = pair_common_friends(bosses, main_data["friends_list"])
            else:
                if filter_type == 'common friends of selected':
                    edges = get_club_unanimity(bosses, main_data["friends_list"])
                else: 
                    if filter_type == "all data (DANGER)":
                        edges = main_data["friends_list"]
                    else:
                        edges = get_friends_in_club(bosses, main_data["friends_list"])



# if filter_type == "All edges":
#     filtered_edges = main_data["friends_list"]
# else:
#     filtered_edges = [
#         (u, v) for (u, v) in main_data["friends_list"]
#         if u in bosses or v in bosses
#     ]



################################################# Styling



st.sidebar.header("Styling",
help="""
You can change some aestetics parameters, like color and size of nodes.
If show images is checked, where available the images are shown.
If images in hd is checked, where available they are in hd.
scale nodes by degree works only for node shapes that do not contains the label, 
so for the shapes: dot, diamond, triangle, triangleDown, square, star.
If is False the size is the value of Node size, 
if is True is the degree of each node multiplied by a scaling factor.
""")


with st.sidebar:

    images_where_available = st.checkbox('node images', False)
    images_in_hd = st.checkbox('node images in hd', False)
    username_label = st.checkbox('node username label', True, help = 'show pop up label')
    edge_label = st.checkbox('edge usernames hover', True)
    directed_graph = st.checkbox('directed graph', False,
help = f'''
If Checked, the edges are shown as arrows, from the follower to the followed.
''')
    differentiate_bridges = st.checkbox('differentiate bridges', True,
help = f'''
Show edges that are bridges or local bridges with a different colour.

from [networx documentation](https://networkx.org/nx-guides/content/exploratory_notebooks/facebook_notebook.html#bridges):

An edge joining two nodes A and B in the graph is considered a bridge, 
if deleting the edge would cause A and B to lie in two different components.

An edge joining two nodes C and Din a graph is a local bridge, 
if its endpoints C and D have no friends in common. 
Very importantly, an edge that is a bridge is also a local bridge.
''')
    #scale_nodes = st.checkbox('scale nodes by degree', False)
    scale_nodes = st.selectbox('nodes scaling method',
    ['node size all equal', 'scale nodes by degree', 'scale nodes by voting degree'])

    if scale_nodes == 'node size all equal':
        node_size = st.slider("Node size", 1, 50, 20, 1)
    else:
        scale_degree = st.slider("degree scaling factor", 0.01, 20.0, 1.0, 0.01)
    
    voting_iterations = st.slider("degree voting iterations", 1, 20, 3, 1)
    
    nodes_shape = st.selectbox('Nodes Shape:', ['dot', 'text', 'circle', 'box', 'ellipse', 'diamond',  'triangle', 'triangleDown', 'square', 'star', 'database'])
    user_A_color = st.color_picker("Main users color", "#DF8404")
    user_B_color = st.color_picker("Interesting users color", "#DB34D5")
    user_C_color = st.color_picker("Other users color", "#3498DB")
    

    edge_color = st.color_picker("Edge color", "#AAAAAA")
    edge_width = st.slider("Edge width", 0.05, 20.0, 2.0, 0.01)

    if differentiate_bridges:
            edge_color_b = st.color_picker("Bridge color", "#CD0D0D")
            edge_width_b = st.slider("Bridge width",0.05, 20.0, 1.5, 0.01)
            edge_color_bl = st.color_picker("Local Bridge color", "#16BD93")
            edge_width_bl = st.slider("Local Bridge width", 0.05, 20.0, 5.0, 0.01)


############################ node labels settings


st.sidebar.header('Show in node\'s label: ',
help = """
Choose the info to show in the nodes labels.
Almost all the info, except for username, are available only
for users labeled as main users or interesting users
(default orange and purple nodes).
""")

show_degree = st.sidebar.checkbox('show nodes degree', True)
show_gb = st.sidebar.checkbox('show nodes global degree', True)
show_iterated_degree = st.sidebar.checkbox('show ivn degree', False)

chiavi = list(main_data['users_info'][users_list[0]].keys())
chiavi_b = {}
start_true = ["full_name", "biography", "followers_count", "following_count", "media_count", "is_private"]
for k in chiavi:
    if k in start_true:
        chiavi_b[k] = st.sidebar.checkbox(f"show {k}", True)
    else:
        chiavi_b[k] = st.sidebar.checkbox(f"show {k}", False)





############################################### advanced settings

st.sidebar.header("Andvanced settings", 
help="""
Check the additional built-in panels of the graph viewer (given by pyvis).
Between them, search menu and filter menu are shown above the graph, the others below.
If You want to have fun maybe You are interested in the physics and interaction panels.
If You want to search a hierarchy of the nodes maybe You want to use the layout panel.
""")

sel_menu = st.sidebar.checkbox("Search menu", value=False)
fil_menu = st.sidebar.checkbox("Filter menu", value=False)
navbuttons = st.sidebar.checkbox("Activate navbuttons", value=True)


# --- Convert to Pyvis ---

if show_graph:
    if darktheme:
        net = Network(height="700px", width="100%", 
                    notebook=False, 
                    directed = directed_graph, 
                    select_menu = sel_menu,
                    filter_menu = fil_menu,
                    cdn_resources = 'remote',
                    bgcolor="#222222", font_color="white")
    else:
        net = Network(height="700px", width="100%", 
                    notebook = False, 
                    directed = directed_graph, 
                    select_menu = sel_menu,
                    filter_menu = fil_menu,
                    cdn_resources = 'remote')



    opt_ui = ['nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer']
    opts_sel = []

    opt_bool = {}

    for tag in opt_ui:
        if tag == 'physics':
            opt_bool[tag] = st.sidebar.checkbox("Set "+ tag, value=True)
        else:
            opt_bool[tag] = st.sidebar.checkbox("Set "+ tag, value=False)


    for tag in opt_ui:
        if opt_bool[tag]:
            opts_sel.append(tag)
        else:
            if tag in opts_sel:
                opts_sel.pop(tag)


    net.show_buttons(filter_ = opts_sel)
    net.toggle_physics(enable_physics)

    options = {
        "interaction": {
            "hover": True,
            "keyboard": {
                "enabled": True
            },
            "navigationButtons": navbuttons
        }
    }

    #net.set_options(json.dumps(options))  #this rewrites the net.options dictionary completely

    # --- Merge, not replace ---
    # Convert current options to dict
    current_options = json.loads(net.options.to_json())

    # Update with your new options
    current_options.update(options)

    # Set back the merged options
    net.set_options(json.dumps(current_options))  #this allows to have both the initial options and the ui panel








################### graph building

G = nx.Graph()
if bosses:
    G.add_edges_from(edges)
    G.add_nodes_from(list(set(nodes + list(G.nodes()))))

    nnodes = G.number_of_nodes()

    degrees = dict(G.degree)
    global_degrees = st.session_state.global_degree

    if (scale_nodes == 'scale nodes by voting degree') or show_iterated_degree:
        ivn_degree = iterated_degree_voting_normalized(nodes, nodes, edges, iter=voting_iterations)


    for node in G.nodes():
        label = node
        title = node

        if show_degree:
            title += f" \n degree : {degrees[node]}"

        if show_gb:
            title += f" \n global degree : {global_degrees[node]}"

        if show_iterated_degree:
            title += f" \n ivnd score : {np.round(ivn_degree[node]* nnodes , 3)}"


        if node in main_data["users_info"]:
            info = main_data["users_info"][node]
            for k in list(info.keys()):
                if chiavi_b[k]:
                    title += f" \n {k} : {info[k]}"
            #title = f"{node}\n{info['username']}, \n {info['biography']}"
        
        G.nodes[node]["label"] = label
        G.nodes[node]["title"] = title
        if scale_nodes == 'scale nodes by degree':   #'scale nodes by degree', 'scale nodes by voting degree'
            G.nodes[node]['size'] = int(degrees[node] * scale_degree)
        elif scale_nodes == 'scale nodes by voting degree':
            G.nodes[node]['size'] = int(ivn_degree[node] * nnodes * scale_degree)
        else:
            G.nodes[node]['size'] = node_size


st.session_state.G = G




####################################### build graph viewer

pos = layout_dict[layout_choice](G)

def add_coloured_node(node, data):
    if node in main_data["users_info"]:
        if main_data["users_info"][node]['scraped_followers']:
            color = user_A_color
        else:
            color = user_B_color
    else:
        color = user_C_color

    if not username_label:
        data["label"] = '   '

    x, y = pos[node]
    net.add_node(
        node,
        label = data["label"],
        title = data['title'],
        x=float(x*1000),
        y=float(y*1000),
        color=color,
        shape = nodes_shape,
        size = data['size'],
        )

def add_ld_image_node(node, data):
    x, y = pos[node]

    if not username_label:
        data["label"] = '   '

    net.add_node(
        node,
        label = data["label"],
        title = data['title'],
        x=float(x*1000),
        y=float(y*1000),
        size = data['size'],
        shape = 'circularImage',
        image = main_data['profilepic'][node]['base64'],
        ) 


def add_hd_image_node(node, data):
    x, y = pos[node]
    if not username_label:
        data["label"] = '   '

    net.add_node(
    node,
    label = data["label"],
    title = data['title'],
    x=float(x*1000),
    y=float(y*1000),
    size = data['size'],
    shape = 'circularImage',
    image = url_to_base64(main_data['profilepic'][node]['urlhd']) if 'urlhd' in main_data['profilepic'][node] else main_data['profilepic'][node]['base64'],
    )


# for node, data in G.nodes(data=True):
    
#     if (node in main_data['profilepic']) and images_where_available:
#         if images_in_hd:
#             add_hd_image_node(node, data)
#         else:
#             add_ld_image_node(node, data)
#     else:
#         add_coloured_node(node, data)


if show_graph:
    if images_where_available:
        
            if images_in_hd:
                for node, data in G.nodes(data=True):
                    if (node in main_data['profilepic']):
                        add_hd_image_node(node, data)
                    else:
                        add_coloured_node(node, data)                
            else:
                for node, data in G.nodes(data=True):
                    if (node in main_data['profilepic']):
                        add_ld_image_node(node, data)  
                    else:
                        add_coloured_node(node, data) 
    else:
        for node, data in G.nodes(data=True):
            add_coloured_node(node, data)



if differentiate_bridges:
    # Identify Bridges (edges whose removal increases the number of connected components)
    # A bridge is also a local bridge, so we should identify bridges first to assign a priority color.
    bridges = list(nx.bridges(G))
    # Identify Local Bridges (edges whose endpoints have no common neighbors, i.e., not part of a triangle)
    # Note: networkx.local_bridges() yields (u, v, span) by default, so we take the first two elements.
    local_bridges = [(u, v) for u, v, span in nx.local_bridges(G, with_span=True)]
    
    if show_graph:

        edlabel = '   '

        # Convert to a set of frozensets for easy and order-independent lookup
        bridges_set = {frozenset(edge) for edge in bridges}
        # Convert to a set of frozensets
        local_bridges_set = {frozenset(edge) for edge in local_bridges}

        for u, v in G.edges():
            frozenedge = frozenset((u, v))

            if edge_label:
                edlabel = f'{u} ---> {v}'

            if frozenedge in bridges_set:
                net.add_edge(u, v, color=edge_color_b, width=edge_width_b, title=edlabel)
            elif frozenedge in local_bridges_set:
                net.add_edge(u, v, color=edge_color_bl, width=edge_width_bl, title=edlabel)
            else:
                net.add_edge(u, v, color=edge_color, width=edge_width, title=edlabel)
else:

    if show_graph:
        edlabel = '     '
        for u, v in G.edges():
            if edge_label:
                edlabel = f'{u} ---> {v}'
            net.add_edge(u, v, color=edge_color, width=edge_width, title=edlabel)





if show_graph:
    # --- Render directly to HTML string ---
    html_str = net.generate_html()


    st.caption(f"Graph Viewer:    this graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # --- Embed in Streamlit ---
    st.components.v1.html(html_str, height=700, scrolling=True)










######################################### graph stats

if show_graph_stats:

    st.header('Graph Stats')

    st.caption('analysis tool for the graph above')

    cc1, cc2, cc3 = st.columns(3)

    with cc1:
        compute_coeff = st.checkbox('Compute Coefficients', True)

    with cc2:
        compute_degree = st.checkbox('Degree Distribution', True)

    with cc3:
        compute_clique = st.checkbox('Count Cliques', False)


    #compute_graph_stats = st.button('Compute Graph Stats')


    if G.edges: #and compute_graph_stats:
        

        if compute_coeff:
            nnodes = G.number_of_nodes()
            nedges = G.number_of_edges()
            maxm = int(comb(nnodes,2))


            st.markdown(f' ------ num of nodes: {nnodes} ------ num of edges: {nedges}')


            st.markdown(f'max number of possible edges: {maxm}',
    help = '''
    It is equal to 
    $${n \\choose 2}$$
    where $n$ is the number of nodes in the graph.
    ''')


            st.markdown(f'graph density: {np.round(nedges/maxm, 3)}',
    help = '''
    It is equal to 
    $$\\frac{m}{{n \\choose 2}} = \\frac{2m}{n(n-1)}$$
    where $n$ is the number of nodes, and $m$ is the number of edges.
    It is the ratio between the actual number of edges and the maximum possible
    given the number of nodes, so it goes from 0 (empty network) to 1 (full network).
    ''')



            st.markdown(f'clustering coefficient: {np.round(nx.average_clustering(G),3)}' , 
    help = '''
    from [networx documentation](https://networkx.org/nx-guides/content/exploratory_notebooks/facebook_notebook.html#clustering-effects):
    The clustering coefficient of a node $v$
    is defined as the probability that two randomly selected friends of $v$
    are friends with each other. 
    As a result, the average clustering coefficient is the average of clustering coefficients of all the nodes. 
    The closer the average clustering coefficient is to 1, 
    the more complete the graph will be because there’s just one giant component. 
    Lastly, it is a sign of triadic closure because the more complete the graph is, 
    the more triangles will usually arise.
    ''')
            if differentiate_bridges:
                nb = len(bridges)
                nbl = len(local_bridges)
            else:
                nb = len(list(nx.bridges(G)))
                nbl = len(list(nx.local_bridges(G)))

            st.markdown(f'num of bridges: {nb}' , 
    help = '''
    from [networx documentation](https://networkx.org/nx-guides/content/exploratory_notebooks/facebook_notebook.html#bridges):
    An edge joining two nodes A and B in the graph is considered a bridge, 
    if deleting the edge would cause A and B to lie in two different components.
    ''')
            

            st.markdown(f'num of local bridges: {nbl}' , 
    help = '''
    from [networx documentation](https://networkx.org/nx-guides/content/exploratory_notebooks/facebook_notebook.html#bridges):
    an edge joining two nodes C and D
    in a graph is a local bridge, if its endpoints C and D
    have no friends in common. 
    Very importantly, an edge that is a bridge is also a local bridge.
    ''')

            st.markdown(f'assortativity coefficient: {np.round(nx.degree_pearson_correlation_coefficient(G),3)}' , 
    help = '''
    from [networx documentation](https://networkx.org/nx-guides/content/exploratory_notebooks/facebook_notebook.html#assortativity):
    Assortativity describes the preference for a network\'s nodes 
    to attach to others that are similar in some way.
    The assortativity coefficient is the 
    Pearson correlation coefficient of degree between pairs of linked nodes. 
    That means that it takes values from -1 to 1. 
    In detail, a positive assortativity coefficient 
    indicates a correlation between nodes of similar degree, 
    while a negative indicates correlation between nodes of different degrees.
    ''')


        if compute_degree:
            degree_array = np.array(list(dict(G.degree).values()))
            st.markdown(f'Degree distribution:  \n mean: {np.round(np.mean(degree_array), 3)} ---- devstd: {np.round(np.std(degree_array), 3)} ---- min: {np.min(degree_array)} ----  max: {np.max(degree_array)} ',
    help = '''
    The degree of a node is the number of its neighbours. 
    So a graph contains as many degree observations as the number of nodes.
    So it presents a distribution of degrees
    that can be analyzed with common descriptive stats.
    ''')


            cd1 , cd2 = st.columns(2)
            degree_df = pd.DataFrame({'nodes degree' : degree_array})

            with cd1:
                fig = px.histogram(degree_df)
                #fig.update_layout({'showlegend': False})
                if darktheme:
                    fig.update_layout({
                    'plot_bgcolor':'black',  # Set the background color here
                    'paper_bgcolor': 'black'
                    })
                st.plotly_chart(fig, use_container_width=True)

            with cd2:
                fig = px.box(degree_df)
                if darktheme:
                    fig.update_layout({
                    'plot_bgcolor':'black',  # Set the background color here
                    'paper_bgcolor': 'black'
                    })
                st.plotly_chart(fig, use_container_width=True)            


        if compute_clique:
            cliques =  nx.enumerate_all_cliques(G)
            mycliques_counts = {}
            for c in cliques:
                k = len(c)
                if k not in mycliques_counts:
                    mycliques_counts[k] = 0
                mycliques_counts[k] +=1

            cliques_str = ''
            for k in mycliques_counts:
                cliques_str += f' --- {k} : {mycliques_counts[k]}'


            st.markdown(f'cliques counts: \n\n {cliques_str}',
    help = '''
    A clique of k nodes is a dense subgraph, so a subgraph where all nodes are connected.
    ''')

            clique_df = pd.DataFrame({'clique nodes': mycliques_counts.keys(), 'counts' : mycliques_counts.values()})
            fig = px.scatter(clique_df, 'clique nodes', 'counts', 
                            title='counts of cliques by dimension').update_traces(mode="lines+markers")
            if darktheme:
                fig.update_layout({
                'plot_bgcolor':'black',  # Set the background color here
                'paper_bgcolor': 'black'
                })
            st.plotly_chart(fig, use_container_width=True)            







# st.header('Global Stats')

# st.caption('distributions of follower and following in the entire db')
