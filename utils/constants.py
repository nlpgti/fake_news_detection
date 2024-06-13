from string import punctuation

from nltk.corpus import stopwords

from utils.utils import re_expressions, re_expressions_independent_position

valid_words = ["i", "am", "of", "ticktock", "ctv", "graham", "putin", "ticktock", "scryp", "isil", "prw", "hittle",
               "islamophobia",
               "ott", "my", "ghosh", "sdk", "islamists", "stalled", "corvette", "to", "in", "rcmpgrc", "forbes",
               "regram", "umer"
    , "stl", "la", "times", "france", "in", "uk", "rt", "deutsch", "vass", "techau", "snowden", "penn", "jillette",
               "sagar", "nyt",
               "belvedere", "wien", "we", "ny", "sponky", "sony", "is", "nsw", "emmerson", "dav"]

filtered_bad_words_ORES = ["a+nus+", "ass+", "bitch", "bootlip", "butt+", "chlamydia", "cholo", "chug", "cocksuck",
                           "coonass", "cracker", "cunt",
                           "dick", "dothead", "(f|ph)ag+(ot)?", "fart", "fat", "fuck", "gipp", "gippo", "gonorrhea",
                           "gook", "gringo", "gypo", "gyppie",
                           "gyppo", "gyppy", "herpes", "hillbilly", "hiv", "hori", "idiot", "injun", "jap", "kike",
                           "kwashi", "kyke",
                           "lick", "motherfuck", "nig", "nig+(a|e|u)+(r|h)+", "niggress", "niglet", "nigor", "nigr",
                           "nigra", "pecker(wood)?",
                           "peni(s)?", "piss", "quashi", "raghead", "redneck", "redskin", "roundeye", "scabies",
                           "shi+t+", "slut", "spi(g|c|k)+",
                           "spigotty", "spik", "spook", "squarehead", "st(u|oo+)pid", "suck", "syphil+is", "turd",
                           "twat", "wank", "wetback", "whore",
                           "wog", "wop", "yank", "yankee", "yid", "zipperhead"]


class TextElements(object):
    word_expanded = ['no', 'mediolargo', 'euro_zona', 'noinflation', 'resisteir',
                     'fundamental+técnico', 'ponderacion', 'shortear', 'sobreponderación', 'atras',
                     'esperadoscae', 'reaccion', 'positividad', 'heroes', 'escierre', 'record', 'preocuante',
                     'break', 'billions', 'seguira', 'analisistecnico', 'leaderboard', 'algocabreados',
                     'stock', 'profecia', 'audiomercadosfuturos', 'prohibe',
                     'contramedida', 'reversar', 'apalancamiento', 'diligence', 'reposicionamiento',
                     'transition', 'torpetrader', 'breaking', 'regression', 'bastar_y',
                     'impresionanti', 'desbalance', 'decision', 'hacer_cargo', 'tradedeal', 'express', 'die',
                     'reversal', 'crack', 'shop', 'populismo', 'interesting', 'superinvestors', 'parabola',
                     'open', 'rapidamente', 'hacer_frente', 'despuésdelcierre', 'subita', 'retailers', 'análise',
                     'intradia', 'sepuedeintentar', 'sesión_orden', 'manipulandiaa', 'mini', 'sectorbancario',
                     'medioplazo', 'compa', 'externalizan', 'fear', 'canibalizacion', 'hacer_referencia',
                     'dto', 'upside', 'posttraumático', 'long', 'dolarizada', 'siguendo', 'abiertaentre',
                     'original_stop', 'players', 'construccion', 'gameoverglobal', 'ummh', 'exposicion', 'informacion',
                     'investing', 'solucion', 'luckin', 'mercadosdevalores', 'bolsadevalores', 'mercado_europeo',
                     'dinàmico', 'empezo', 'updateden', 'lafuriadewallstreet', 'fallir', 'desactualizar',
                     'posibilida', 'preopen', 'financial', 'target', 'raised', 'subira', 'shock', 'closing',
                     'rating', 'overbought', 'competitive', 'muchoo', 'versus', 'tading', 'vengahastaluego',
                     'capitaldeinter', 'technicalanalysis', 'continues', 'zonavalue', 'sobrecapacidad', 'futures',
                     'bursátil_bueno', 'siguebajando', 'swingtrading', 'antimonopolio', 'siemptealcistas', 'minicompra',
                     'good', 'mayoria', 'intradialo', 'educacionfinanciera', 'higher', 'enfrio', 'invertironline',
                     'bullmarket', 'dato_macro', 'sintomas', 'lamentablee', 'espectativas', 'trollear',
                     'consolidacion', 'shorteados', 'shorteaste', 'sotniene', 'cercaria', 'inverores',
                     'maldito_especulador', 'continue', 'esperadoreacción', 'enlaapertura',
                     'equivoquép', 'intradiario', 'banca_seguro', 'bolsamania', 'falls', 'subvaluar',
                     'alertastopindice', 'explosion', 'resultado_banco', 'sellers', 'punto_futuro', 'apertura_bajista',
                     'dividendera', 'sinceramientofalso', 'stress', 'monitoreando', 'daxsuperación', 'dividendopia',
                     'volvera', 'aceercarse', 'rallie', 'vendededores', 'infraponderar', 'paciencia_y', 'masdividendos',
                     'cronica', 'faciles', 'asurtar', 'ends', 'esplayando', 'cierremercadosemanal',
                     'proximas', 'posteriori', 'dividendc', 'near', 'hold', 'esperadosube', 'maximo', 'refinanciar',
                     'mantenia', 'muchoinvertir', 'lowcost', 'sobreventa', 'debil',
                     'macro+', 'insanity', 'rebound', 'upstream', 'frenaria', 'stochfish',
                     'c_dividendo', 'dailysemana', 'buff', 'sostenibilidad', 'atencion', 'wars', 'stop', 'updated',
                     'buenoo', 'below', 'precausion', 'noticia_seguro', 'garantias', 'guerracomercial', 'throwback',
                     'priceado', 'impresion', 'informacionnofinanciera', 'retestear', 'jajajajalong', 'enqueinvertirla',
                     'traumatica', 's_ubio', 'risk', 'muysufrida', 'optimissa', 'invesion', 'ostión', 'semana_bajista',
                     'arruinamania', 'inversión_inteligente', 'acumulacion', 'sobrecompra', 'bearmarket', 'g_o',
                     'hitting', 'esperadosingresos', 'expansion', 'adquisitions', 'agusto', 'wrong', 'tradewar',
                     'confimar', 'deploman', 'crackpd', 'attenti', 'panico', 'beyond', 'runnerinversor',
                     'desinversión', 'ultimos', 'throwbacks', 'incomer', 'future', 'resfria', 'par_principal', 'solid',
                     'woow',
                     'volvio', 'ahh', 'strongbullmarket', 'reuperará', 'maximos', 'bullmarketsco', 'tradereturns',
                     'buyandhold',
                     'deploma', 'proyeccion', 'traders', 'subian', 'short', 'largo_corto', 'small', 'increased',
                     'preocupacion', 'belico', 'recogidadebeneficios', 'especulaciondecortoplazo', 'rates', 'growth',
                     'advertian',
                     'business', 'above', 'close', 'party', 'riesgopais', 'adverti', 'yield', 'performance',
                     'tontera_bolsera', 'rompio', 'análisis_técnico', 'shortearla', 'ahora_bien', 'hypergrowth',
                     'optimismocomercial',
                     'stake', 'equity', 'aceleracion', 'lomejor', 'super', 'leaderpullback', 'strong', 'offer',
                     'increiblee',
                     'mrdividendo', 'encontar', 'reinversion', 'cost', 'olgadamente', 'caidas', 'uptick', 'dividend',
                     'ultrabajo',
                     'pullback', 'ingresaria', 'gmbeneficio', 'riesgoreputacional', 'contra_tendencia', 'ganaria',
                     'querramos', 'upar', 'attaque', 'beto', 'influencer', 'top', 'atención_a', 'sobrecomprar',
                     'confirmaria', 'consolidates', 'closed', 'corto_rabioso', 'maxima', 'potenciaria', 'breakeven',
                     'previsionsemanal', 'cerrarian', 'guerracomercialel', 'fake', 'recesion', 'intradiaria',
                     'sell', 'quebro', 'poors', 'loaded', 'profit', 'cierranenrojo', 'largo_plazo', 'reacelerar',
                     'plazofijo', 'pocoo', 'supercharger', 'intraday', 'unicas', 'poor', 'comprá', 'renegociar',
                     'asesoramiento_financiero', 'líder_crecimiento', 'selling', 'recuperacion', 'cuts',
                     'consolidating', 'lider', 'tailored', 'booking', 'reevaluar', 'preajuste', 'break_out',
                     'huge', 'o_ver', 'inferiror', 'caera', 'invasion', 'fact', 'desempleovie', 'descending',
                     'valoresaseguir',
                     'highs', 'prinicpales', 'certidumbre+productividad+crecimiento', 'lagar_de', 'subiendocomo',
                     'enjoy',
                     'buybacks', 'puts', 'intradía', 'tedencias', 'bagger', 'bajar_caída', 'shorted', 'anticompetencia',
                     'tendencial', 'correccion', 'take', 'tradewars', 'afinanza', 'guerracomercialy', 'minimos',
                     'conquesttrades',
                     'coming', 'crazy', 'bu_ll', 'deep', 'inadvertir', 'bear', 'caida', 'sorpresivamente',
                     'tradear', 'sobreponderar', 'investor', 'corto_medio', 'bancoscaen', 'waiting', 'caer_e',
                     'acuerdocomercial', 'gates', 'ranking', 'pulsodemercado', 'advanced', 'gone', 'tonteos', 'bonus',
                     'mercado_alcista', 'historico']
    list_chars_not_to_remove = ['¿', '‼️', '¡', '!', "!", "?", "+", "-"]
    spanish_stopwords_add = ["telefónica", "telefónico", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
                             "agosto", "septiembre",
                             "octubre",
                             "noviembre", "diciembre", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado",
                             "domingo",
                             "a", "además", "ahí", "anual", "aquí", "argentino", "así", "aun", "aunque", "cada",
                             "centilitro",
                             "chino", "cualquiera", "cómo", "de", "día", "dónde", "el", "español", "etcétera",
                             "europeo", "hoy", "le", "lo", "me", "milímetro", "país", "pero",
                             "pues", "se", "semana", "semanal", "semestre", "te",
                             "trimestral", "trimestre", "índice", "nos", "centilitro"]

    list_word_exceptions = ["+", "ticker", "stock", "abbr", "currency", "material", "num", "numneg", "numpos",
                            "proyección", "proyectar", "date"]

    list_chars_to_remove = ['#', '@', '\'', '´', "º", "ª", "°", "\"", "{", "}", "<", ">", ";", "=", "_", "-", "`", "~",
                            "%",
                            "/",
                            ":", "*",
                            "(", ")", "|", "\\n", "€", "\\", "&", "[", "]", ",", ".", "?", "¿", "!", "¡"]

    list_emojis_acepted = ["↗", "↘", "⏫", "☑", "✅", "⬆", "🆗", "🆙", "🌟", "🍾", "🏆", "👌", "👌🏽", "👍", "👍🏼",
                           "👍🏾",
                           "👎", "👏", "👏🏻",
                           "💣", "💪", "💪🏼", "💰", "💲", "💴", "🔝", "🔟", "🔥", "🔪", "😀", "😁", "😂", "😃", "😄",
                           "😎",
                           "😏", "😐",
                           "😕", "😘", "😜", "😨", "😫", "😭", "😰", "😻", "🙂", "🚀", "🛑", "🛫", "🤑", "🤔", "🤢",
                           "🤣",
                           "🤦", "🤩",
                           "🤪", "🤬", "🤮", "🤯", "🥰", "🥳"]

    non_words = list(punctuation).extend(map(str, range(10)))
    spanish_stopwords = [elem for elem in stopwords.words('spanish') if not (
                elem in ['no', 'si', 'sí', 'más', 'pero', 'muy', 'sin', "mucha", "muchas", "mucho", 'muchos', 'poco',
                         'pocos', 'nada'])]

    alphabet = ["a", "e", "i", "o", "u", "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t",
                "v", "w", "x", "z"]

    re_spanish_stopwords = re_expressions(spanish_stopwords)

    re_list_chars_to_remove = re_expressions_independent_position(list_chars_to_remove)
