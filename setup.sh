mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"alexander.verheecke@hotmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
<<<<<<< HEAD
[general]\n\
email = \"alexander.verheecke@hotmail.com\"\n\
" > ~/.streamlit/secrets.toml

echo "\
=======
>>>>>>> parent of e5b2e39 (secrets setup.sh)
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
<<<<<<< HEAD
" > ~/.streamlit/config.toml

# echo "\
# [server]\n\
# headless = true\n\
# enableCORS=false\n\
# port = $PORT\n\
# " > ~/.streamlit/secrets.toml
=======
" > ~/.streamlit/config.toml
>>>>>>> parent of e5b2e39 (secrets setup.sh)
