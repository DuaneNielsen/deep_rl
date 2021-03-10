Search.setIndex({docnames:["buffer","capture","checkpoint","config","distributions","driver","env","gymviz","index","modules","observer","pong_dataset","wandb_utils"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["buffer.rst","capture.rst","checkpoint.rst","config.rst","distributions.rst","driver.rst","env.rst","gymviz.rst","index.rst","modules.rst","observer.rst","pong_dataset.rst","wandb_utils.rst"],objects:{"":{buffer:[0,0,0,"-"],capture:[1,0,0,"-"],checkpoint:[2,0,0,"-"],config:[3,0,0,"-"],distributions:[4,0,0,"-"],driver:[5,0,0,"-"],env:[6,0,0,"-"],gymviz:[7,0,0,"-"],observer:[10,0,0,"-"],pong_dataset:[11,0,0,"-"],wandb_utils:[12,0,0,"-"]},"buffer.DiscountedReturns":{step:[0,2,1,""]},"buffer.Enricher":{enrich:[0,2,1,""],reset:[0,2,1,""],step:[0,2,1,""]},"buffer.FullTransition":{a:[0,2,1,""],d:[0,2,1,""],i:[0,2,1,""],r:[0,2,1,""],s:[0,2,1,""],s_p:[0,2,1,""]},"buffer.ReplayBuffer":{clear:[0,2,1,""],enrich:[0,2,1,""],reset:[0,2,1,""],step:[0,2,1,""],tail_trajectory_complete:[0,2,1,""]},"buffer.Returns":{step:[0,2,1,""]},"capture.JpegCapture":{done:[1,2,1,""],reset:[1,2,1,""],step:[1,2,1,""]},"capture.PngCapture":{done:[1,2,1,""],reset:[1,2,1,""],step:[1,2,1,""]},"capture.VideoCapture":{done:[1,2,1,""],reset:[1,2,1,""],step:[1,2,1,""]},"config.ArgumentParser":{add_argument:[3,2,1,""],parse_args:[3,2,1,""]},"config.NullScheduler":{step:[3,2,1,""]},"distributions.ScaledTanhTransformedGaussian":{entropy:[4,2,1,""],enumerate_support:[4,2,1,""],mean:[4,2,1,""],rsample:[4,2,1,""],sample:[4,2,1,""],variance:[4,2,1,""]},"distributions.TanhTransform":{atanh:[4,2,1,""],bijective:[4,4,1,""],codomain:[4,4,1,""],domain:[4,4,1,""],log_abs_det_jacobian:[4,2,1,""],sign:[4,4,1,""]},"distributions.TanhTransformedGaussian":{entropy:[4,2,1,""],enumerate_support:[4,2,1,""],mean:[4,2,1,""],variance:[4,2,1,""]},"env.continuous_cartpole":{ContinuousCartPoleEnv:[6,1,1,""]},"env.continuous_cartpole.ContinuousCartPoleEnv":{close:[6,2,1,""],metadata:[6,4,1,""],render:[6,2,1,""],reset:[6,2,1,""],seed:[6,2,1,""],step:[6,2,1,""],stepPhysics:[6,2,1,""]},"env.debug":{Bandit:[6,1,1,""],DelayedBandit:[6,1,1,""],DummyEnv:[6,1,1,""],LineGrid:[6,1,1,""],LinearEnv:[6,1,1,""],StaticEnv:[6,1,1,""],one_hot:[6,3,1,""]},"env.debug.DummyEnv":{reset:[6,2,1,""],step:[6,2,1,""]},"env.debug.LineGrid":{d_s:[6,2,1,""],done:[6,2,1,""],lookahead:[6,2,1,""],render:[6,2,1,""],reset:[6,2,1,""],reward:[6,2,1,""],step:[6,2,1,""]},"env.debug.LinearEnv":{done:[6,2,1,""],render:[6,2,1,""],reset:[6,2,1,""],reward:[6,2,1,""],step:[6,2,1,""]},"env.debug.StaticEnv":{reset:[6,2,1,""],step:[6,2,1,""]},"env.wrappers":{ActionBranches:[6,1,1,""],AddDoneToState:[6,1,1,""],ApplyFunc:[6,1,1,""],AtariAriVector:[6,1,1,""],ClipState2D:[6,1,1,""],ConcatDeltaPrev:[6,1,1,""],ConcatPrev:[6,1,1,""],FireResetEnv:[6,1,1,""],FrameStack:[6,1,1,""],LazyFrames:[6,1,1,""],MaxAndSkipEnv:[6,1,1,""],NoopResetEnv:[6,1,1,""],Normalizer:[6,1,1,""],RemapActions:[6,1,1,""],Resize2D:[6,1,1,""],RewardCountLimit:[6,1,1,""],RewardOneIfNotDone:[6,1,1,""],TimeLimit:[6,1,1,""]},"env.wrappers.ActionBranches":{reset:[6,2,1,""],step:[6,2,1,""]},"env.wrappers.AddDoneToState":{reset:[6,2,1,""],step:[6,2,1,""]},"env.wrappers.ApplyFunc":{observation:[6,2,1,""]},"env.wrappers.AtariAriVector":{extract:[6,2,1,""],reset:[6,2,1,""],step:[6,2,1,""]},"env.wrappers.ClipState2D":{observation:[6,2,1,""]},"env.wrappers.ConcatDeltaPrev":{reset:[6,2,1,""],step:[6,2,1,""]},"env.wrappers.ConcatPrev":{reset:[6,2,1,""],step:[6,2,1,""]},"env.wrappers.FireResetEnv":{reset:[6,2,1,""],step:[6,2,1,""]},"env.wrappers.FrameStack":{reset:[6,2,1,""],step:[6,2,1,""]},"env.wrappers.LazyFrames":{astype:[6,2,1,""]},"env.wrappers.MaxAndSkipEnv":{reset:[6,2,1,""],step:[6,2,1,""]},"env.wrappers.NoopResetEnv":{reset:[6,2,1,""],step:[6,2,1,""]},"env.wrappers.RemapActions":{action:[6,2,1,""]},"env.wrappers.Resize2D":{observation:[6,2,1,""]},"env.wrappers.RewardCountLimit":{reset:[6,2,1,""],step:[6,2,1,""]},"env.wrappers.RewardOneIfNotDone":{step:[6,2,1,""]},"env.wrappers.TimeLimit":{reset:[6,2,1,""],step:[6,2,1,""]},"gymviz.Plot":{reset:[7,2,1,""],save:[7,2,1,""],step:[7,2,1,""],write_video:[7,2,1,""]},"observer.EnvObserver":{reset:[10,2,1,""],step:[10,2,1,""]},"observer.StateCapture":{done:[10,2,1,""],reset:[10,2,1,""],step:[10,2,1,""]},"observer.SubjectWrapper":{append_step_filter:[10,2,1,""],attach_observer:[10,2,1,""],detach_observer:[10,2,1,""],observe_step:[10,2,1,""],observer_reset:[10,2,1,""],reset:[10,2,1,""],step:[10,2,1,""]},"wandb_utils.LogRewards":{reset:[12,2,1,""],step:[12,2,1,""]},buffer:{DiscountedReturns:[0,1,1,""],Enricher:[0,1,1,""],FullTransition:[0,1,1,""],ReplayBuffer:[0,1,1,""],ReplayBufferDataset:[0,1,1,""],Returns:[0,1,1,""],TrajectoryTransitions:[0,1,1,""],TrajectoryTransitionsReverse:[0,1,1,""],wrap:[0,3,1,""]},capture:{JpegCapture:[1,1,1,""],PngCapture:[1,1,1,""],VideoCapture:[1,1,1,""]},checkpoint:{load:[2,3,1,""],sample_policy_returns:[2,3,1,""],save:[2,3,1,""]},config:{ArgumentParser:[3,1,1,""],NullScheduler:[3,1,1,""],counter:[3,3,1,""],exists_and_not_none:[3,3,1,""],flatten:[3,3,1,""],get_kwargs:[3,3,1,""],get_optim:[3,3,1,""],set_if_not_set:[3,3,1,""]},distributions:{ScaledTanhTransformedGaussian:[4,1,1,""],TanhTransform:[4,1,1,""],TanhTransformedGaussian:[4,1,1,""]},driver:{episode:[5,3,1,""],render_env:[5,3,1,""],step_environment:[5,3,1,""]},env:{continuous_cartpole:[6,0,0,"-"],debug:[6,0,0,"-"],wrappers:[6,0,0,"-"]},gymviz:{Cooldown:[7,1,1,""],Plot:[7,1,1,""]},observer:{EnvObserver:[10,1,1,""],RewardFilter:[10,1,1,""],StateCapture:[10,1,1,""],StepFilter:[10,1,1,""],SubjectWrapper:[10,1,1,""]},wandb_utils:{LogRewards:[12,1,1,""],nancheck:[12,3,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:attribute"},terms:{"01603":4,"1912":4,"255":6,"652r":6,"break":6,"case":[0,1,6,7,10,12],"class":[0,1,3,4,6,7,10,12],"default":3,"float":[0,1,6,7,10,12],"function":[0,1,5,6,7,10,12],"int":6,"new":[0,1,6,7,10,12],"return":[0,1,2,3,4,6,7,10,12],"static":4,"super":6,"switch":3,"true":[0,4,5,6],And:6,The:[3,4,6],With:4,abs:4,accept:[0,1,6,7,10,12],accident:6,action:[0,1,5,6,7,10,12],actionbranch:6,actionwrapp:6,add:6,add_argu:3,adddonetost:6,added:0,adding:10,advanc:5,affinetransform:4,after:[0,1,6,7,10,12],agent:[0,1,6,7,10,12],alia:0,all:[4,6],along:4,alreadi:3,amount:[0,1,6,7,10,12],ani:6,ansi:6,append_step_filt:10,appli:[2,4,6],applyfunc:6,aren:6,arg:[0,1,3,4,6,7,10,12],argpars:3,argument:[2,3],argumentpars:3,arrai:6,arxiv:4,associ:2,astyp:6,atanh:4,atariarivector:6,attach:[0,10],attach_observ:10,attr:3,automat:6,auxiliari:[0,1,6,7,10,12],bandit:6,base:[0,1,3,4,6,7,10,12],batch:[0,4],batch_shap:4,becom:3,befor:10,behavior:4,being:[0,3,4],best:2,best_optim:2,best_polici:2,between:[0,1,4,6,7,10,12],bigint:6,biject:4,book:6,bool:[0,1,4,6,7,10,12],box:6,buffer:[8,9],c9zm:6,cache_s:4,calcul:0,call:[0,1,5,6,7,10,12],can:[3,6],captur:[6,8,9],cardin:4,cart:6,cartesian:4,cartpol:7,cartpolecontinu:3,check:3,checkpoint:[8,9],classic:6,cleanup:6,clear:0,clipstate2d:6,close:6,code:[3,6],codomain:4,collat:0,collect:[0,6],color:6,come:4,command:3,comment:3,compat:3,complet:0,composetransform:4,comput:4,concatdeltaprev:6,concatprev:6,config:[8,9],configur:3,construct:3,consumpt:6,contain:[0,1,3,4,6,7,10,12],content:[8,9],continu:6,continuous_cartpol:9,continuouscartpoleenv:6,control:4,conveni:0,convent:6,cooldown:7,copi:6,core:[0,1,6,7,10,12],correl:6,counter:3,creat:[2,3],current:[0,1,3,6,7,10,12],d_s:6,danforth:6,data:[0,10],debug:[0,1,7,9,10,12],def:6,delai:5,delayedbandit:6,descript:3,det:4,detach_observ:10,determinist:6,devic:10,diagnost:[0,1,6,7,10,12],dict:[0,1,3,6,7,10,12],dim:4,dimens:4,directori:[1,2,3],discount:0,discountedreturn:0,discret:4,disk:2,displai:6,distribut:[8,9],doesn:6,doesnt:2,domain:4,done:[0,1,6,7,10,12],dream:4,driver:[8,9],dtype:6,dummyenv:6,dure:0,dynam:[0,1,6,7,10,12],each:[0,1,5,6,7,10,12],effici:0,elif:6,els:[3,6],empti:[0,3],end:[0,1,6,7,10,12],enrich:[0,10],ensur:[4,6],entropi:4,enumer:4,enumerate_support:4,env:[0,1,2,3,5,7,8,9,10,12],environ:[0,1,2,5,6,7,10,12],envobserv:10,episod:[0,1,5,6,7,10,12],episodes_per_batch:3,episodes_per_point:7,equal:6,equival:4,error:12,escap:6,event_shap:4,everi:4,exampl:6,except:6,exist:2,exists_and_not_non:3,exit:6,expand:4,extract:6,fals:[0,2,4,5],field:0,file:[2,3],filter:10,fireresetenv:6,first:[4,6],flatten:3,float16:6,forc:6,frame:6,frame_op:6,frame_op_len:6,frames_per_second:6,framestack:6,from:[0,2,3,4,6],full:4,fulltransit:0,func:6,further:[0,1,6,7,10,12],garbag:6,gaussian:4,gener:[0,1,4,5,6,7,10,12],get_kwarg:3,get_optim:3,given:[0,4,6],gym:[0,1,5,6,7,10,12],gymviz:[8,9],hand:10,happen:4,has:[0,1,6,7,10,12],hello:3,help:[0,1,6,7,10,12],high:6,history_length:7,howev:4,http:[4,6],human:6,hyphen:3,i_p:0,ian:6,imag:6,imagin:4,implement:[6,10],includ:6,incompleteidea:6,independ:[0,1,6,7,10,12],index:8,inf:4,info:[0,1,6,7,10,12],info_kei:0,inform:[0,1,6,7,10,12],inital_st:6,initi:[0,1,6,7,10,12],initial_st:6,input:[4,5],instead:4,interfac:10,interpol:6,interv:4,iter:[0,4,5],itertool:4,jacobian:4,jpegcaptur:1,just:6,keep:3,kei:[0,3,6],kwarg:[0,2,3,4,5,6,10],label:6,last:6,latent:4,lazyfram:6,learn:[0,1,4,6,7,10,12],length:[7,12],line:3,linearenv:6,linegrid:6,list:[0,3,4,6],load:[0,2,3],lock:4,log:[4,12],log_abs_det_jacobian:4,logreward:12,look:2,lookahead:6,low:6,lower_bound:4,main:6,make:[6,7],map:4,mass:4,match:4,max:[4,6],max_episode_step:6,max_reward_count:6,maxandskipenv:6,mean:4,mean_return:2,mess:6,metadata:6,method:[0,2,6],metric:12,might:4,min:[4,6],mode:6,model:3,modul:[8,9],most:[0,3],multipl:[0,1,6,7,10,12],must:5,myenv:6,n_state:6,name:[2,3,10,12],namespac:3,nan:4,nancheck:12,ndarrai:6,necessari:6,nest:3,net:6,new_spac:6,newlin:6,noframeskip:6,none:[0,2,3,6,7,10,12],noop_max:6,noopresetenv:6,normal:6,note:[0,1,4,6,7,10,12],noth:6,nullschedul:3,num_color:6,number:[0,1,6,7,10,12],numer:4,numpi:6,object:[0,1,3,6,7,10,12],observ:[0,1,5,6,7,8,9,12],observationwrapp:6,observe_step:10,observer_reset:10,often:6,one:[0,1,4,5,6,7,10,12],one_hot:6,openai:6,oper:0,optim:[2,3],option:3,order:[0,6],org:4,other:[0,1,6,7,10,12],otherwis:3,out:2,output:[4,5,6],output_observation_spac:6,over:[0,4,6],overrid:6,packag:[8,9],page:8,param:[3,5,6,12],paramet:[3,4],parent_kei:3,pars:3,parse_arg:3,parser:3,pass:[3,5,6,10],per:[6,7],perform:6,perma:6,permalink:6,pixel:6,place:3,placehold:3,plot:7,pluggabl:10,pngcaptur:1,pole:6,polici:[2,5,10],policy_net:2,pong_dataset:[8,9],pop:6,possibl:10,pre:10,precend:3,prefix:[2,12],prepend:12,preprocess:10,prevent:6,previou:[0,1,6,7,10,12],prime:0,probabl:4,process:[0,10],product:4,program:6,properti:[0,4],provid:[0,1,5,6,7,10,12],pseudorandom:6,rais:6,random:[0,1,6,7,10,12],reach:[0,1,6,7,10,12],read:[0,3],real:4,recent:[0,3],recommend:[4,6],refresh_cooldown:7,remain:4,remap:6,remapact:6,render:[2,5,6],render_env:5,reparameter:4,repeat:6,replai:0,replay_buff:0,replaybuff:0,replaybufferdataset:0,repres:6,represent:6,reproduc:6,reset:[0,1,6,7,10,12],resize2d:6,respons:[0,1,6,7,10,12],result:[0,1,4,6,7,10,12],retriev:0,revers:0,reward:[0,1,6,7,10,12],reward_map:6,rewardcountlimit:6,rewardfilt:10,rewardoneifnotdon:6,rgb:6,rgb_arrai:6,rich:6,rsampl:4,run:[0,1,2,3,5,6,7,10,12],run_42:2,run_id:3,runnabl:5,s_p:0,sampl:[0,1,2,4,6,7,10,12],sample_policy_return:2,sample_shap:4,satur:4,save:[2,7],scale:4,scaledtanhtransform:4,scaledtanhtransformedgaussian:4,schedul:3,search:8,sec:7,seed:[3,6,10],self:6,sep:3,separ:3,sequenc:6,set:[3,6],set_if_not_set:3,shape:[4,6],should:[0,1,4,6,7,10,12],sigmoidtransform:4,sign:4,simpli:6,singl:[0,5],singleton:4,size:6,skip:6,skipfram:6,some:6,sometim:[0,1,6,7,10,12],space:6,specifi:3,stabl:4,stack:6,start:0,state:[0,1,5,6,7,10,12],state_dict:2,state_prepro:10,statecaptur:10,staticenv:6,stdev_return:2,step:[0,1,3,4,5,6,7,10,12],step_environ:5,stepfilt:10,stepphys:6,store:3,str:6,string:6,stringio:6,style:6,subclass:6,subjectwrapp:10,submodul:9,subtract:6,suitabl:[0,1,6,7,10,12],sum:6,support:[4,6,10],sure:6,sutton:6,system:6,tail_trajectory_complet:0,take:[2,5],taken:[2,3],tanh:4,tanhtransform:4,tanhtransformedgaussian:4,tensor:[4,12],termin:6,text:6,thei:0,them:10,themselv:6,thi:[0,1,4,6,7,10,12],thu:4,timelimit:6,timestep:[0,1,6,7,10,12],titl:7,torch:[2,4],total:0,track:3,trajectori:[0,2,6],trajectory_start_end_tupl:0,trajectorytransit:0,trajectorytransitionsrevers:0,transform:4,transformed_distribut:4,transformeddistribut:4,transit:[0,5],tupl:[0,1,6,7,10,12],turn:6,uint8:6,undefin:[0,1,6,7,10,12],univari:4,upper_bound:4,use:[3,4,5,6,10],used:[0,6,10],uses:2,using:[2,5],usual:6,valu:[2,3,4,6],vari:6,variabl:[0,1,6,7,10,12],varianc:4,version:6,via:4,video:6,videocaptur:1,wai:0,wandb:12,wandb_util:[8,9],want:[6,10],weight:2,were:0,when:[0,1,4,6,7,10,12],where:4,whether:[0,1,4,6,7,10,12],which:[0,1,3,6,7,10,12],window:6,won:6,word:[0,1,6,7,10,12],world:3,wrap:0,wrapper:[0,1,7,9,10,12],write:2,write_video:7,yaml:3,yield:[0,1,6,7,10,12],you:[0,1,6,7,10,12],your:6},titles:["buffer module","capture module","checkpoint module","config module","distributions module","driver module","env package","gymviz module","Welcome to deep_rl\u2019s documentation!","deep_rl","observer module","pong_dataset module","wandb_utils module"],titleterms:{buffer:0,captur:1,checkpoint:2,config:3,content:6,continuous_cartpol:6,debug:6,deep_rl:[8,9],distribut:4,document:8,driver:5,env:6,gymviz:7,indic:8,modul:[0,1,2,3,4,5,6,7,10,11,12],observ:10,packag:6,pong_dataset:11,submodul:6,tabl:8,wandb_util:12,welcom:8,wrapper:6}})